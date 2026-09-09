"""Full-batch compiled CUDA checks for each newly tested width."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save
from looping.width.common import SOURCE_PATHS, build_model, protocol, run_config, validate_data, verify_reference
from looping.width.evaluate import evaluate_arrays, export_model, load_export
from looping.weight_tying.train import restore_rng, rng_state
from runtime_utils import runtime_manifest

def gpu_preflight(data_dir, output_dir, arm):
    if not torch.cuda.is_available():
        raise RuntimeError("This preflight requires CUDA")
    verify_reference()
    started = time.perf_counter()
    output_dir = Path(output_dir)
    atomic_json_save(runtime_manifest(SOURCE_PATHS), output_dir / "environment.json")
    data_identity = validate_data(data_dir)
    with np.load(Path(data_dir) / "train.npz", allow_pickle=False) as arrays:
        digits, targets = arrays["digits"][:2048], arrays["targets"][:2048]
    inputs = F.one_hot(torch.as_tensor(digits, device="cuda").long(), 10).float()
    answers = torch.as_tensor(targets, device="cuda").long()
    torch.set_float32_matmul_precision("high")
    results = []
    for arm in (arm,):
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = run_config(arm, protocol()["seeds"][0])
        torch.manual_seed(config["seed"])
        model = build_model(config).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
        print(f"PREFLIGHT_CONFIG arm={arm} width={model.width} batch=2048 compiled=True", flush=True)
        forward, advance = torch.compile(model), torch.compile(model.advance)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            early_loss, _, _ = forward(inputs, answers)
        early_loss.backward()
        print("PREFLIGHT_EARLY_BACKWARD_DONE", flush=True)
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

        # Compile both entry signatures before testing restored RNG and optimizer state.
        late_step()
        print("PREFLIGHT_LATE_BACKWARD_DONE", flush=True)
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
        result = {"arm": arm, "width": model.width, "parameters": sum(p.numel() for p in model.parameters()),
                  "early_loss": float(early_loss), "late_loss": late_loss,
                  "populated_optimizer_resume_exact": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "state_diagnostics": diagnostic}
        results.append(result)
        print("PREFLIGHT_ARM " + json.dumps(result, sort_keys=True), flush=True)
        del model, optimizer, forward, advance, loaded, saved, actual, expected
    result = {"status": "passed", "batch_size": 2048, "compiled": True, "evaluation_iterations": [16, 128, 1024, 4096], "supervised_iterations": 16,
              "prefix_iterations": 512, "arms": results, "data_sha256": data_identity,
              "source_sha256": runtime_manifest(SOURCE_PATHS)["source_sha256"],
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json_save(result, output_dir / "result.json")
    return result
