"""Reference equivalence, gate behavior, resumability, and full-batch CUDA checks."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save
from inference import RecurrentRunner
from looping.hyperloop.analyze import analyze, compare
from looping.hyperloop.common import SOURCE_PATHS, build_model, protocol, run_config, state_sha256, validate_data
from looping.hyperloop.evaluate import evaluate_arrays, evaluate_run, export_model, load_export
from looping.hyperloop.model import HyperloopTransformer, LoopGates
from looping.hyperloop.train import train_run
from looping.weight_tying.common import atomic_npz
from looping.weight_tying.test_study import fixture, make_smoke_data
from looping.weight_tying.train import restore_rng, rng_state
from model_io import load_model
from runtime_utils import file_sha256, runtime_manifest
from stabilize.exp_testbed_20k import SudokuTransformer


class HyperloopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_baseline_matches_public_model_loss_gradients_and_dropout(self):
        inputs, answers = fixture(2)
        torch.manual_seed(21)
        reference = SudokuTransformer()
        torch.manual_seed(21)
        baseline = HyperloopTransformer()
        self.assertEqual(state_sha256(reference.state_dict()), state_sha256(baseline.state_dict()))
        for training in (False, True):
            reference.train(training)
            baseline.train(training)
            reference.zero_grad(set_to_none=True)
            baseline.zero_grad(set_to_none=True)
            torch.manual_seed(19)
            logits = reference(inputs, return_all=True)
            mask = inputs[:, :, 0]
            expected = torch.stack([(F.cross_entropy(value.flatten(0, 1), answers.flatten(), reduction="none")
                                     .reshape_as(mask) * mask).sum() / mask.sum() for value in logits]).mean()
            torch.manual_seed(19)
            actual, last, _ = baseline(inputs, answers)
            torch.testing.assert_close(last, logits[-1], rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.backward()
            expected.backward()
            for left, right in zip(baseline.parameters(), reference.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_base_initialization_is_paired_and_extra_parameters_are_small(self):
        digests = []
        for streams in (0, 1, 4):
            torch.manual_seed(71)
            model = HyperloopTransformer(streams)
            digests.append(state_sha256({name: tensor for name, tensor in model.state_dict().items()
                                        if not name.startswith("gates.")}))
            self.assertLess(sum(p.numel() for p in model.parameters()), 805000)
        self.assertEqual(len(set(digests)), 1)

    def test_gate_equation_and_parallel_streams_are_not_independent_transformers(self):
        inputs, _ = fixture(2)
        model = HyperloopTransformer(4).eval()
        hidden, probabilities = model.initial_state(inputs)
        self.assertEqual(hidden.shape, (2, 81, 4, 128))
        self.assertEqual(len(model.layers), 4)
        read, write, retain = model.gates(hidden)
        combined = (hidden * read.unsqueeze(-1)).sum(-2)
        proposal = model.recurrent_step(combined, probabilities, model.rope_cos, model.rope_sin)
        expected = hidden * retain.unsqueeze(-1) + proposal.unsqueeze(-2) * write.unsqueeze(-1)
        actual, _, logits = model.step(hidden, probabilities)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(logits, model.output_head(actual.mean(-2)), rtol=0, atol=0)
        self.assertGreater(float((actual[:, :, 0] - actual[:, :, 1]).abs().max().detach()), 0)
        for values, upper in ((read, 0.5), (write, 2), (retain, 1)):
            self.assertTrue(bool(((values > 0) & (values < upper)).all()))

    def test_gate_normalization_does_not_modify_the_carried_state(self):
        gates = LoopGates(4)
        hidden = torch.randn(2, 81, 4, 128) * 20
        before = hidden.clone()
        gates(hidden)
        torch.testing.assert_close(hidden, before, rtol=0, atol=0)
        first, second = gates(hidden), gates(hidden * 2)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right)

    def test_failed_and_large_states_have_serializable_diagnostics(self):
        model = HyperloopTransformer(4)
        for magnitude in (float("inf"), float("nan"), 1e30):
            hidden = torch.full((1, 81, 4, 128), magnitude)
            json.dumps(model.state_diagnostics(hidden), allow_nan=False)

    def test_every_gate_receives_gradients_after_detached_prefix(self):
        inputs, targets = fixture(2)
        for streams in (1, 4):
            model = HyperloopTransformer(streams).eval()
            with torch.no_grad():
                state = model.advance(*model.initial_state(inputs))
            self.assertTrue(all(not value.requires_grad for value in state))
            loss, _, _ = model(inputs, targets, initial_state=state)
            loss.backward()
            self.assertIsNone(model.initial_encoder.weight.grad)
            for name, parameter in model.gates.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(parameter.grad.isfinite().all()), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_chunked_prefix_matches_direct_recurrence(self):
        inputs, targets = fixture()
        for streams in (0, 1, 4):
            model = HyperloopTransformer(streams).eval()
            with torch.no_grad():
                state = model.initial_state(inputs)
                chunked = model.advance(*model.advance(*state))
                for _ in range(32):
                    hidden, probabilities, _ = model.step(*state)
                    state = hidden, probabilities
                for left, right in zip(chunked, state):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                loss, _, _ = model(inputs, targets, initial_state=state)
                other_targets = targets.clone()
                other_targets[:, 1:] = (other_targets[:, 1:] + 1) % 9
                changed, _, _ = model(inputs, other_targets, initial_state=state)
                torch.testing.assert_close(loss, changed, rtol=0, atol=0)

    def test_evaluation_matches_public_runner_and_restores_training_precision(self):
        inputs, targets = fixture(3)
        baseline = HyperloopTransformer().train()
        reference = SudokuTransformer()
        reference.load_state_dict(baseline.state_dict())
        expected, diagnostics = RecurrentRunner(reference, track_solutions=True).run_batch(
            inputs, [1, 16, 32], targets=targets, empty_mask=inputs[:, :, 0].bool())
        precision = torch.get_float32_matmul_precision()
        scores, actual, _ = evaluate_arrays(baseline, inputs.argmax(-1).numpy(), targets.numpy(),
                                            [1, 16, 32], batch_size=3, track_solutions=True)
        self.assertTrue(baseline.training)
        self.assertEqual(torch.get_float32_matmul_precision(), precision)
        for horizon, logits in expected.items():
            np.testing.assert_array_equal(actual[f"predictions_{horizon}"], logits.argmax(-1).numpy())
            self.assertEqual(scores[str(horizon)]["total"], 3)
        for name in ("first_solved_iteration", "regression_count", "ever_solved", "stayed_solved_after_first"):
            np.testing.assert_array_equal(actual[name], diagnostics[name].numpy())

    def test_resume_is_exact_for_all_arms_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            results = []
            for arm in protocol()["arms"]:
                seed = protocol()["seeds"][0]
                directory = root / arm
                train_run(root / "data", directory, arm, seed, smoke=True, stop_after=2, device="cpu")
                # Simulate preemption between the selected export and checkpoint publication.
                atomic_torch_save({}, directory / "best_validation.pt")
                resumed = train_run(root / "data", directory, arm, seed, smoke=True, device="cpu")
                whole = train_run(root / "data", root / (arm + "_whole"), arm, seed, smoke=True, device="cpu")
                results.append(resumed)
                self.assertEqual(resumed["sample_digest"], whole["sample_digest"])
                self.assertEqual(state_sha256(torch.load(directory / "final.pt", weights_only=True)),
                                 state_sha256(torch.load(root / (arm + "_whole") / "final.pt", weights_only=True)))
                loaded, manifest = load_export(directory / "final.pt")
                _, selected = load_export(directory / "best_validation.pt")
                self.assertEqual(selected["updates"], resumed["best_validation"]["updates"])
                self.assertEqual(loaded.streams, protocol()["arms"][arm])
                self.assertEqual(manifest["updates"], 4)
                with self.assertRaises(ValueError):
                    load_model(directory / "final.pt")
                with self.assertRaises(ValueError):
                    train_run(root / "data", directory, arm, seed + 1, smoke=True, device="cpu")
                altered = torch.load(directory / "final.pt", weights_only=True)
                altered["initial_encoder.bias"][0] += 1
                atomic_torch_save(altered, directory / "final.pt")
                with self.assertRaises(ValueError):
                    load_export(directory / "final.pt")
            for result in results[1:]:
                for key in ("sample_digest", "base_initial_state_sha256", "work_counts"):
                    self.assertEqual(result[key], results[0][key])

    def test_failed_seeds_cannot_disappear_from_decision(self):
        self.assertEqual(compare({})["status"], "pending")
        profiles = {(arm, seed): {"1024": 0.94, "4096": 0.9}
                    for arm in protocol()["arms"] for seed in protocol()["seeds"]}
        self.assertEqual(compare(profiles)["status"], "not_promising")
        for seed in protocol()["seeds"]:
            profiles["gated_four", seed] = {"1024": 0.945, "4096": 0.90}
        self.assertEqual(compare(profiles)["status"], "promising")
        profiles["gated_four", protocol()["seeds"][0]] = None
        self.assertEqual(compare(profiles)["status"], "not_promising")

    def test_full_evaluation_metadata_selection_reuse_and_independent_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            spec = protocol()
            benchmark = json.loads((Path(__file__).parents[2] / spec["evaluation"]["benchmark"]).read_text())
            inputs, answers = fixture()
            digits = np.repeat(inputs.argmax(-1).numpy(), 25000, axis=0)
            targets = np.repeat(answers.numpy(), 25000, axis=0)
            labels, indices = np.array(benchmark["bucket_names"]), np.array(benchmark["indices"])
            atomic_npz(data_dir / "development.npz", digits=digits, targets=targets, labels=labels, indices=indices)
            data_identity = {"development.npz": file_sha256(data_dir / "development.npz")}
            config = run_config("baseline", spec["seeds"][0])
            source_identity = runtime_manifest(SOURCE_PATHS)["source_sha256"]
            directory = root / "runs" / f"baseline_seed{spec['seeds'][0]}"
            directory.mkdir(parents=True)
            model = build_model(config)
            for selection, step in (("final", 20000), ("best_validation", 19000)):
                export_model(model, directory / f"{selection}.pt", config, step, data_identity, source_identity)
            training = {"config": config, "status": "complete", "updates": 20000,
                        "source_sha256": source_identity, "data_sha256": data_identity,
                        "best_validation": {"updates": 19000}, "parameters": 796937,
                        "base_initial_state_sha256": "fixture", "sample_digest": "fixture", "work_counts": {},
                        "timings_seconds": {}, "history": [{"updates": 20000, "scores": {"1024": {"accuracy": 1.0}}}]}
            atomic_json_save(training, directory / "result.json")
            scores, predictions = {}, {}
            for horizon in spec["evaluation"]["iterations"]:
                predictions[f"predictions_{horizon}"] = targets.astype(np.uint8)
                predictions[f"solved_{horizon}"] = np.ones(25000, dtype=bool)
                predictions[f"finite_{horizon}"] = np.ones(25000, dtype=bool)
                scores[str(horizon)] = {"accuracy": 1.0, "solved": 25000, "total": 25000, "nonfinite": 0}
            with patch("looping.hyperloop.evaluate.validate_data", return_value=data_identity), \
                 patch("looping.hyperloop.common.validate_data", return_value=data_identity), \
                 patch("looping.hyperloop.evaluate.evaluate_arrays", return_value=(scores, predictions, {})) as evaluator:
                for selection in ("final", "best_validation"):
                    result = evaluate_run(data_dir, directory, selection, device="cpu")
                    self.assertEqual(result["identity"]["checkpoint_selection"], selection)
                    self.assertEqual(result["identity"]["selection"], spec["evaluation"]["selection"])
                    self.assertEqual(evaluate_run(data_dir, directory, selection, device="cpu"), result)
                self.assertEqual(evaluator.call_count, 2)
                report = analyze(root, data_dir)
                self.assertEqual(report["versus_baseline"]["status"], "pending")
                output = directory / "evaluations/final"
                predictions["predictions_1024"] = targets.copy()
                predictions["predictions_1024"][0, 0] = (targets[0, 0] + 1) % 9
                atomic_npz(output / "predictions.npz", **predictions)
                changed = json.loads((output / "result.json").read_text())
                changed["predictions_sha256"] = file_sha256(output / "predictions.npz")
                atomic_json_save(changed, output / "result.json")
                with self.assertRaises(AssertionError):
                    analyze(root, data_dir)


def gpu_preflight(data_dir, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("This preflight requires CUDA")
    started = time.perf_counter()
    output_dir = Path(output_dir)
    data_identity = validate_data(data_dir)
    with np.load(Path(data_dir) / "train.npz", allow_pickle=False) as arrays:
        digits, targets = arrays["digits"][:2048], arrays["targets"][:2048]
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95))
        forward, advance = torch.compile(model), torch.compile(model.advance)
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

        # Compile both entry signatures before testing restored RNG and optimizer state.
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
        scores, _, diagnostic = evaluate_arrays(loaded, digits[:8], targets[:8], [16, 128, 1024], batch_size=8)
        if any(score["nonfinite"] for score in scores.values()):
            raise ValueError(f"Nonfinite FP32 inference: {arm}")
        result = {"arm": arm, "parameters": sum(p.numel() for p in model.parameters()),
                  "early_loss": float(early_loss), "late_loss": late_loss,
                  "populated_optimizer_resume_exact": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "state_diagnostics": diagnostic}
        results.append(result)
        print("PREFLIGHT_ARM " + json.dumps(result, sort_keys=True), flush=True)
        del model, optimizer, forward, advance, loaded, saved, actual, expected
    result = {"status": "passed", "batch_size": 2048, "supervised_iterations": 16,
              "prefix_iterations": 512, "arms": results, "data_sha256": data_identity,
              "source_sha256": runtime_manifest(SOURCE_PATHS)["source_sha256"],
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json_save(result, output_dir / "result.json")
    return result


if __name__ == "__main__":
    unittest.main()
