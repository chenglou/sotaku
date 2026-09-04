"""Selection, recurrence equivalence, replay, resumption, and CUDA preflight."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save
from looping.weight_tying.test_study import fixture, make_smoke_data
from looping.weight_tying.train import PairedSampler, restore_rng, rng_state
from looping.window_selection.analyze import decision, validate_evaluation
from looping.window_selection.common import SOURCE_PATHS, protocol, run_config, validate_data
from looping.window_selection.selection import WindowTransformer, confidence_score, scan_candidates, select_index
from looping.window_selection.train import export, train_run
from model_io import load_model
from runtime_utils import runtime_manifest
from stabilize.exp_testbed_20k import SudokuTransformer


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selection_rules_and_invalid_scores(self):
        self.assertEqual(select_index([0.5, 0.9, 0.8], "confidence", 0), 1)
        self.assertEqual(select_index([0.9, 0.9], "confidence", 1), 0)
        self.assertEqual(select_index([0.9, 0.1], "random", 1), 1)
        self.assertEqual(select_index([0.9, 0.1], "latest", 0), 1)
        for scores in ([], [float("nan")], [float("inf")]):
            with self.assertRaises((ValueError, FloatingPointError)):
                select_index(scores, "confidence", 0)
        with self.assertRaises(ValueError):
            run_config("bogus", protocol()["seeds"][0])

    def test_confidence_masks_givens_and_averages_all_iterations(self):
        logits = torch.zeros(16, 2, 81, 9)
        mask = torch.zeros(2, 81, dtype=torch.bool)
        mask[:, 0] = True
        logits[:, :, 1:, 0] = 100
        self.assertAlmostEqual(float(confidence_score(logits, mask)), 1 / 9, places=6)
        logits[:8, :, 0, 0] = 100
        self.assertAlmostEqual(float(confidence_score(logits, mask)), (1 + 1 / 9) / 2, places=6)

    def test_model_loss_and_gradient_match_reference(self):
        inputs, answers = fixture(2)
        torch.manual_seed(84)
        reference = SudokuTransformer().eval()
        model = WindowTransformer().eval()
        model.load_state_dict(reference.state_dict())
        logits = reference(inputs, return_all=True)
        mask = inputs[:, :, 0]
        expected = torch.stack([(F.cross_entropy(value.flatten(0, 1), answers.flatten(), reduction="none")
                                 .reshape_as(mask) * mask).sum() / mask.sum() for value in logits]).mean()
        actual, _, replay = model(inputs, answers)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(replay, torch.stack(logits), rtol=0, atol=0)
        expected.backward()
        actual.backward()
        for left, right in zip(reference.parameters(), model.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_exact_cpu_dropout_replay_and_detachment(self):
        inputs, answers = fixture(2)
        model = WindowTransformer().train()
        scan = scan_candidates(model, inputs, model.scan_window, (16, 32))
        self.assertTrue(all(not value.requires_grad for state in scan.states for value in state))
        for index in range(2):
            restore_rng(scan.rng_states[index])
            loss, _, replay = model(inputs, answers, initial_state=scan.states[index])
            torch.testing.assert_close(replay, scan.logits[index], rtol=0, atol=0)
            loss.backward()
            self.assertIsNone(model.initial_encoder.weight.grad)
            self.assertTrue(all(parameter.grad is not None for layer in model.layers for parameter in layer.parameters()))
            model.zero_grad(set_to_none=True)
            restore_rng(scan.end_rng)
            torch.testing.assert_close(torch.get_rng_state(), scan.end_rng["torch"], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            scan_candidates(model, inputs, model.scan_window, (17,))

    def test_scan_boundaries_match_direct_recurrence(self):
        inputs, _ = fixture()
        model = WindowTransformer().eval()
        scan = scan_candidates(model, inputs, model.scan_window, (16, 32))
        with torch.no_grad():
            state = model.initial_state(inputs)
            state, _ = model.scan_window(*state)
            for actual, expected in zip(scan.states[0], state):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state, _ = model.scan_window(*state)
            for actual, expected in zip(scan.states[1], state):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sampler_streams_do_not_depend_on_selection(self):
        spec = protocol()
        regime = {"burnin_probability": spec["late_probability"], "burnin_iterations": spec["candidate_starts"]}
        samplers = [PairedSampler(np.array([0, 1, 11, 51]), 32, regime, spec["training"]) for _ in range(3)]
        for step in range(40):
            batches = [sampler.sample(step, 8) for sampler in samplers]
            for batch in batches[1:]:
                np.testing.assert_array_equal(batch[0], batches[0][0])
                self.assertEqual(batch[1], batches[0][1])

    def test_resume_export_and_mismatched_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            seed = protocol()["seeds"][0]
            train_run(root / "data", root / "resume", "confidence", seed,
                      smoke=True, stop_after=2, device="cpu")
            resumed = train_run(root / "data", root / "resume", "confidence", seed, smoke=True, device="cpu")
            whole = train_run(root / "data", root / "whole", "confidence", seed, smoke=True, device="cpu")
            self.assertEqual(resumed["sample_digest"], whole["sample_digest"])
            self.assertEqual(resumed["selection_history"], whole["selection_history"])
            for key, value in torch.load(root / "whole/final.pt", weights_only=True).items():
                torch.testing.assert_close(value, torch.load(root / "resume/final.pt", weights_only=True)[key], rtol=0, atol=0)
            load_model(root / "whole/final.pt")
            with self.assertRaises(ValueError):
                train_run(root / "data", root / "resume", "latest", seed, smoke=True, device="cpu")

    def test_predeclared_decision_keeps_all_seeds(self):
        settings = protocol()
        self.assertEqual(decision({})["status"], "pending")
        profiles = {(selector, seed): {"1024": 0.94, "4096": 0.9}
                    for selector in settings["selectors"] for seed in settings["seeds"]}
        self.assertEqual(decision(profiles)["status"], "not_promising")
        for seed in settings["seeds"]:
            profiles["confidence", seed] = {"1024": 0.95, "4096": 0.9}
        self.assertEqual(decision(profiles)["status"], "promising")
        profiles["confidence", settings["seeds"][0]] = None
        self.assertEqual(decision(profiles)["status"], "not_promising")

    def test_paired_runs_match_initialization_batches_and_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            results = [train_run(root / "data", root / selector, selector, protocol()["seeds"][0],
                                 smoke=True, device="cpu") for selector in protocol()["selectors"]]
            for result in results[1:]:
                for key in ("initial_state_sha256", "sample_digest", "work_counts"):
                    self.assertEqual(result[key], results[0][key])

    def test_evaluation_rejects_changed_precision_and_small_dataset(self):
        settings = protocol()["evaluation"]
        benchmark = json.loads((Path(__file__).parents[2] / settings["benchmark"]).read_text())
        evaluation = {"identity": {"benchmark_rows_sha256": benchmark["rows_sha256"],
                      "iterations": settings["iterations"], "precision": "fp32",
                      "matmul_precision": "highest", "batch_size": 256,
                      "compiled": False, "track_solutions_every_iteration": True},
                      "scores": {str(h): {"solved": 20000, "total": 25000}
                                 for h in settings["iterations"]}}
        validate_evaluation(evaluation)
        changed = copy.deepcopy(evaluation)
        changed["identity"]["precision"] = "bf16"
        with self.assertRaises(ValueError):
            validate_evaluation(changed)
        changed = copy.deepcopy(evaluation)
        changed["scores"]["1024"]["total"] = 1000
        with self.assertRaises(ValueError):
            validate_evaluation(changed)


def gpu_smoke(data_dir, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("The preflight must run on CUDA")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_identity = validate_data(data_dir)
    with np.load(Path(data_dir) / "train.npz", allow_pickle=False) as arrays:
        inputs = F.one_hot(torch.as_tensor(arrays["digits"][:2048], device="cuda").long(), 10).float()
        answers = torch.as_tensor(arrays["targets"][:2048], device="cuda").long()
    torch.manual_seed(20260904)
    model = WindowTransformer().cuda().train()
    assert sum(parameter.numel() for parameter in model.parameters()) == 796937
    settings = protocol()["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
    forward = torch.compile(model)
    advance = torch.compile(model.scan_window)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    print("Compiling and checking an ordinary full-batch update", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _, _ = forward(inputs, answers)
    loss.backward()
    assert model.initial_encoder.weight.grad is not None
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())
    optimizer.step()
    model.zero_grad(set_to_none=True)
    print("Scanning all five candidate windows at batch size 2048", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scan = scan_candidates(model, inputs, advance, protocol()["candidate_starts"])
    comparisons = []
    for selector in protocol()["selectors"]:
        print(f"Checking compiled replay: {selector}", flush=True)
        selected = select_index(scan.scores, selector, 0)
        restore_rng(scan.rng_states[selected])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, replay = forward(inputs, answers, initial_state=scan.states[selected])
        assert loss.isfinite()
        loss.backward()
        assert model.initial_encoder.weight.grad is None
        assert all(parameter.grad is not None and parameter.grad.isfinite().all()
                   for name, parameter in model.named_parameters() if not name.startswith("initial_encoder."))
        comparisons.append({"selector": selector, "selected_start": scan.starts[selected],
                            "scan_confidence": scan.scores[selected],
                            "replay_confidence": float(confidence_score(replay.detach(), inputs[:, :, 0].bool())),
                            "maximum_logit_difference": float((replay.detach() - scan.logits[selected]).abs().max())})
        restore_rng(scan.end_rng)
        del replay, loss
        if selector != protocol()["selectors"][-1]:
            model.zero_grad(set_to_none=True)
    del scan
    model.eval()
    torch.set_float32_matmul_precision("highest")
    config = run_config("confidence", protocol()["seeds"][0], smoke=True)
    export(model, output_dir / "export.pt", config, 1, data_identity)
    loaded, _ = load_model(output_dir / "export.pt", device="cuda")
    with torch.no_grad():
        _, _, expected = model(inputs[:2], answers[:2])
        torch.testing.assert_close(loaded(inputs[:2]), expected[-1], rtol=0, atol=0)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()}
    atomic_torch_save(state, output_dir / "resume.pt")
    saved = torch.load(output_dir / "resume.pt", map_location="cpu", weights_only=False)
    loaded.load_state_dict(saved["model"])
    restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=0.123)
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert len(restored_optimizer.state) == len(list(loaded.parameters()))
    assert all(int(state["step"]) == 1 for state in restored_optimizer.state.values())
    for original, restored in zip(model.parameters(), loaded.parameters()):
        restored.grad = None if original.grad is None else original.grad.detach().clone()
    optimizer.step()
    restored_optimizer.step()
    for original, restored in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
    restore_rng(saved["rng"])
    expected_random = torch.rand(64, device="cuda")
    restore_rng(saved["rng"])
    torch.testing.assert_close(torch.rand(64, device="cuda"), expected_random, rtol=0, atol=0)
    torch.cuda.synchronize()
    result = {"status": "passed", "data_sha256": data_identity,
              "source_sha256": runtime_manifest(SOURCE_PATHS)["source_sha256"],
              "batch_size": 2048, "candidate_starts": protocol()["candidate_starts"],
              "parameters": 796937, "comparisons": comparisons,
              "populated_optimizer_resume_exact": True,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json_save(result, output_dir / "passed.json")
    print("PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    unittest.main()
