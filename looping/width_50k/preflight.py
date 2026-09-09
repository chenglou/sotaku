"""Small-batch CUDA check, backed by unchanged full-batch compiled preflight evidence."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save
from looping.width_50k.common import SOURCE_PATHS, build_model, protocol, run_config, validate_data, verify_inherited_preflight
from looping.width_50k.evaluate import evaluate_arrays, export_model, load_export
from looping.weight_tying.train import restore_rng, rng_state
from runtime_utils import runtime_manifest


def gpu_preflight(data_dir, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("This preflight requires CUDA")
    started = time.perf_counter()
    inherited = verify_inherited_preflight()
    batch_size = protocol()["preflight"]["additional_cuda_batch_size"]
    output_dir = Path(output_dir)
    atomic_json_save(runtime_manifest(SOURCE_PATHS), output_dir / "environment.json")
    data_identity = validate_data(data_dir)
    with np.load(Path(data_dir) / "train.npz", allow_pickle=False) as arrays:
        digits, targets = arrays["digits"][:batch_size], arrays["targets"][:batch_size]
    inputs = F.one_hot(torch.as_tensor(digits, device="cuda").long(), 10).float()
    answers = torch.as_tensor(targets, device="cuda").long()
    torch.set_float32_matmul_precision("high")
    results = []
    for arm in protocol()["arms"]:
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = run_config(arm, protocol()["seeds"][0])
        torch.manual_seed(config["seed"])
        model = build_model(config).cuda().train()
        print(f"PREFLIGHT_CONFIG arm={arm} width={model.width} batch={batch_size} "
              "compiled=False inherited_compiled_batch=2048", flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
        forward, advance = model, model.advance
        with torch.autocast("cuda", dtype=torch.bfloat16):
            early_loss, _, _ = forward(inputs, answers)
        early_loss.backward()
        if not torch.isfinite(early_loss) or not all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()):
            raise ValueError(f"Nonfinite early gradients: {arm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()},
                          output_dir / f"{arm}_resume.pt")

        def late_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = model.initial_state(inputs)
                for _ in range(32):
                    state = advance(*state)
            if any(value.requires_grad for value in state):
                raise ValueError("Gradient-free prefix retained an autograd graph")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = forward(inputs, answers, initial_state=state)
            loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()):
                raise ValueError(f"Nonfinite late gradients: {arm}")
            if model.gates is not None and any(p.grad is None for p in model.gates.parameters()):
                raise ValueError("A gate is disconnected from the loss")
            optimizer.step()
            return float(loss)

        # Exercise both entry signatures before testing restored RNG and optimizer state.
        late_step()
        saved = torch.load(output_dir / f"{arm}_resume.pt", map_location="cuda", weights_only=False)
        for attempt in range(2):
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            restore_rng({"torch": saved["rng"]["torch"].cpu(), "cuda": [value.cpu() for value in saved["rng"]["cuda"]]})
            late_loss = late_step()
            actual = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if attempt == 0:
                expected = actual
            else:
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        path = output_dir / f"{arm}_export.pt"
        export_model(model, path, config, 2, data_identity, runtime_manifest(SOURCE_PATHS)["source_sha256"])
        loaded, _ = load_export(path, device="cuda")
        scores, _, diagnostic = evaluate_arrays(loaded, digits[:8], targets[:8], [16, 128, 1024, 4096], batch_size=8)
        if any(score["nonfinite"] for score in scores.values()):
            raise ValueError(f"Nonfinite FP32 inference: {arm}")
        result = {"arm": arm, "parameters": sum(p.numel() for p in model.parameters()),
                  "early_loss": float(early_loss), "late_loss": late_loss,
                  "populated_optimizer_resume_exact": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "state_diagnostics": diagnostic}
        results.append(result)
        print("PREFLIGHT_ARM " + json.dumps(result, sort_keys=True), flush=True)
        del model, optimizer, forward, advance, loaded, saved, actual, expected
    result = {"status": "passed", "batch_size": batch_size, "compiled": False, "supervised_iterations": 16,
              "prefix_iterations": 512, "arms": results, "data_sha256": data_identity,
              "source_sha256": runtime_manifest(SOURCE_PATHS)["source_sha256"],
              "protocol": protocol(), "inherited_full_batch_verified": True,
              "inherited_full_batch": {"report_sha256": protocol()["preflight"]["inherited_full_batch_sha256"],
                                       "batch_size": inherited["batch_size"], "compiled": True},
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json_save(result, output_dir / "result.json")
    return result
