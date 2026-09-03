"""Unit tests and the full-batch CUDA preflight for the weight-tying study."""

import copy
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save
from dataset_utils import DATASET_REVISION
from looping.weight_tying.common import SOURCE_PATHS, atomic_npz, protocol, protocol_sha256, run_config
from looping.weight_tying.data import canonical_digits, encode_rows, parse_qqwing, symmetry_keys, validate_solution
from looping.weight_tying.evaluate import evaluate_arrays, load_export, seal_cohort
from looping.weight_tying.model import StudyTransformer, forward_flops_per_iteration, parameter_count, rope_tables
from looping.weight_tying.train import PairedSampler, export_model, learning_rate, train_run
from runtime_utils import file_sha256, runtime_manifest
from stabilize.exp_testbed_20k import ROPE_COS, ROPE_SIN, SudokuTransformer

ANSWER = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
QUESTION = "." + ANSWER[1:]


def fixture(batch_size=1, device="cpu"):
    digits, targets = encode_rows([QUESTION] * batch_size, [ANSWER] * batch_size)
    inputs = F.one_hot(torch.as_tensor(digits, device=device).long(), 10).float()
    return inputs, torch.as_tensor(targets, device=device).long()


def make_smoke_data(directory):
    directory.mkdir(parents=True, exist_ok=True)
    digits, targets = encode_rows([QUESTION] * 32, [ANSWER] * 32)
    atomic_npz(directory / "train.npz", digits=digits, targets=targets, ratings=np.zeros(32, dtype=np.int16))
    atomic_npz(directory / "validation.npz", digits=digits[:2], targets=targets[:2], labels=np.array(["fixture"] * 2))
    atomic_json_save({"protocol_sha256": protocol_sha256(), "dataset_revision": DATASET_REVISION,
                      "files": {name: file_sha256(directory / name) for name in ("train.npz", "validation.npz")}},
                     directory / "manifest.json")


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_and_arithmetic_matching(self):
        models = {key: StudyTransformer(**value) for key, value in protocol()["architectures"].items()}
        self.assertEqual(parameter_count(models["tied"]), 796937)
        self.assertEqual(parameter_count(models["untied_compute"]), 12693257)
        self.assertEqual(parameter_count(models["untied_parameters"]), 797385)
        self.assertLess(abs(parameter_count(models["tied"]) - parameter_count(models["untied_parameters"]))
                        / parameter_count(models["tied"]), 0.001)
        self.assertEqual(forward_flops_per_iteration(128, 512), forward_flops_per_iteration(128, 512))
        self.assertLess(forward_flops_per_iteration(32, 124), forward_flops_per_iteration(128, 512))
        cosine, sine = rope_tables(128)
        torch.testing.assert_close(cosine, ROPE_COS)
        torch.testing.assert_close(sine, ROPE_SIN)

    def test_reference_and_untied_initial_outputs_match(self):
        inputs, targets = fixture()
        torch.manual_seed(23)
        reference = SudokuTransformer().eval()
        torch.manual_seed(23)
        tied = StudyTransformer().eval()
        torch.manual_seed(23)
        untied = StudyTransformer(period=16).eval()
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, tied.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            reference_logits = reference(inputs)
            tied_loss, tied_logits = tied(inputs, targets)
            untied_loss, untied_logits = untied(inputs, targets)
        torch.testing.assert_close(reference_logits, tied_logits, rtol=0, atol=0)
        torch.testing.assert_close(tied_logits, untied_logits, rtol=0, atol=0)
        torch.testing.assert_close(tied_loss, untied_loss, rtol=0, atol=0)

    def test_tied_gradients_equal_sum_of_untied_gradients(self):
        inputs, targets = fixture()
        torch.manual_seed(24)
        tied = StudyTransformer(width=16, feedforward_width=32, dropout=0).double().eval()
        torch.manual_seed(24)
        untied = StudyTransformer(width=16, feedforward_width=32, period=16, dropout=0).double().eval()
        tied(inputs.double(), targets)[0].backward()
        untied(inputs.double(), targets)[0].backward()
        for index in range(4):
            for name, parameter in tied.layers[index].named_parameters():
                total = sum(dict(untied.layers[4 * stage + index].named_parameters())[name].grad
                            for stage in range(16))
                torch.testing.assert_close(parameter.grad, total, rtol=1e-7, atol=1e-8)
        optimizer = torch.optim.AdamW(untied.parameters(), lr=0.002)
        optimizer.step()
        self.assertFalse(torch.equal(untied.layers[0].q_proj.weight, untied.layers[4].q_proj.weight))

    def test_untied_storage_and_repeat_boundary(self):
        model = StudyTransformer(period=16)
        self.assertNotEqual(model.layers[0].q_proj.weight.data_ptr(), model.layers[4].q_proj.weight.data_ptr())
        inputs, _ = fixture()
        hidden, predictions = model.initial_state(inputs)
        with self.assertRaisesRegex(ValueError, "stage 17"):
            model.step(hidden, predictions, 16)
        with torch.no_grad():
            model.step(hidden, predictions, 16, repeat=True)

    def test_masked_loss_ignores_givens(self):
        inputs, targets = fixture()
        model = StudyTransformer(width=16, feedforward_width=32, dropout=0).eval()
        other = targets.clone()
        other[:, 1:] = (other[:, 1:] + 1) % 9
        torch.testing.assert_close(model(inputs, targets)[0], model(inputs, other)[0])

    def test_sampler_is_paired_and_resumable(self):
        settings = protocol()["training"]
        regime = protocol()["regimes"]["late"]
        ratings = np.array([0, 1, 11, 51, 60])
        first = PairedSampler(ratings, 42, regime, settings)
        second = PairedSampler(ratings, 42, regime, settings)
        for step in range(10):
            torch.rand(31)
            left, right = first.sample(step, 64), second.sample(step, 64)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1], right[1])
        state = first.state_dict()
        expected = first.sample(4000, 64)
        second.load_state_dict(state)
        actual = second.sample(4000, 64)
        np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

    def test_data_encoding_and_symmetry(self):
        validate_solution(QUESTION, ANSWER)
        encoded, targets = encode_rows([QUESTION], [ANSWER])
        self.assertEqual(encoded[0, 0], 0)
        self.assertEqual(targets[0, 0], 4)
        renamed = QUESTION.translate(str.maketrans("123456789", "987654321"))
        self.assertEqual(canonical_digits(QUESTION), canonical_digits(renamed))
        rotated = "".join(np.rot90(np.array(list(renamed)).reshape(9, 9)).flat)
        self.assertIn(canonical_digits(rotated), symmetry_keys(QUESTION))
        with self.assertRaises(ValueError):
            validate_solution(QUESTION, ANSWER[::-1])

    def test_generator_requires_unique_solution(self):
        header = "Puzzle,Solution,Solution Count,Difficulty,\n"
        rows = parse_qqwing(header + f"{QUESTION},{ANSWER},1,Simple,\n", "simple")
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            parse_qqwing(header + f"{QUESTION},{ANSWER},2,Simple,\n", "simple")

    def test_lr_and_selection_gate(self):
        settings = protocol()["training"]
        self.assertAlmostEqual(learning_rate(559, settings), 0.002)
        self.assertGreater(learning_rate(19999, settings), 0)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Cohort incomplete"):
                seal_cohort(temporary)
            self.assertFalse((Path(temporary) / "cohort_lock.json").exists())

    def test_cpu_resume_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            partial = train_run(root / "data", root / "resumed", "untied_parameters", "early", 20260902,
                                smoke=True, stop_after=2, device="cpu")
            self.assertEqual(partial["status"], "paused")
            resumed = train_run(root / "data", root / "resumed", "untied_parameters", "early", 20260902,
                                smoke=True, device="cpu")
            uninterrupted = train_run(root / "data", root / "whole", "untied_parameters", "early", 20260902,
                                      smoke=True, device="cpu")
            self.assertEqual(resumed["sample_digest"], uninterrupted["sample_digest"])
            left = torch.load(root / "resumed/final.pt", weights_only=True)
            right = torch.load(root / "whole/final.pt", weights_only=True)
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            loaded, metadata = load_export(root / "resumed/final.pt", device="cpu")
            self.assertEqual(metadata["updates"], 4)
            self.assertEqual(loaded.period, 16)

    def test_late_training_detaches_only_the_input_encoder(self):
        model = StudyTransformer(width=16, feedforward_width=32, period=16)
        inputs, targets = fixture(2)
        loss, _ = model(inputs, targets)
        loss.backward()
        check_training_gradients(model, detached_input=False)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            initial = model.advance(*model.initial_state(inputs))
        loss, _ = model(inputs, targets, initial_state=initial)
        loss.backward()
        check_training_gradients(model, detached_input=True)
        model.output_head.weight.grad = None
        with self.assertRaisesRegex(ValueError, "output_head.weight"):
            check_training_gradients(model, detached_input=True)


def check_training_gradients(model, *, detached_input):
    for name, parameter in model.named_parameters():
        if detached_input and name.startswith("initial_encoder."):
            if parameter.grad is not None:
                raise ValueError(f"Unexpected gradient through detached input encoder: {name}")
        elif parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise ValueError(f"Missing or nonfinite gradient: {name}")


def gpu_smoke(output_dir):
    """Exercise actual 2048-puzzle memory use and both compiled training paths."""
    output_dir = Path(output_dir)
    runtime = runtime_manifest(SOURCE_PATHS)
    source_digest = hashlib.sha256(json.dumps(runtime["source_sha256"], sort_keys=True).encode()).hexdigest()
    destination = output_dir / source_digest
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination / "result.json"
    if completed.exists():
        return json.loads(completed.read_text())
    measurements = {}
    for architecture, settings in protocol()["architectures"].items():
        torch.manual_seed(20260902)
        model = StudyTransformer(**settings).cuda()
        compiled = torch.compile(model)
        advance = torch.compile(model.advance)
        optimizer = torch.optim.AdamW(compiled.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
        inputs, targets = fixture(2048, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for horizon in (0, 32, 512):
            optimizer.zero_grad(set_to_none=True)
            initial = None
            if horizon:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    initial = model.initial_state(inputs)
                    for _ in range(horizon // 16):
                        initial = advance(*initial)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, logits = compiled(inputs, targets, initial_state=initial)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite CUDA smoke loss: {architecture}/{horizon}")
            loss.backward()
            check_training_gradients(model, detached_input=bool(horizon))
            optimizer.step()
            torch.cuda.synchronize()
            print(f"GPU SMOKE {architecture} horizon={horizon} loss={loss.item():.6f}", flush=True)
        checkpoint = destination / f"{architecture}_resume.pt"
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        restored = StudyTransformer(**settings).cuda()
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
        restored.load_state_dict(saved["model"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(saved["optimizer"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        measurements[architecture] = {"parameters": parameter_count(model),
                                      "elapsed_seconds": time.perf_counter() - started,
                                      "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "compiled_base_and_late_training": True,
                                      "optimizer_checkpoint_loaded": True}
        del compiled, advance, optimizer, model, restored, restored_optimizer, inputs, targets, saved, initial, loss, logits
        torch.cuda.empty_cache()
    result = {"status": "passed", "protocol_sha256": protocol_sha256(), "source_sha256": runtime["source_sha256"],
              "runtime": runtime, "measurements": measurements}
    atomic_json_save(result, completed)
    atomic_json_save({"path": str(completed), "sha256": file_sha256(completed),
                      "source_sha256": runtime["source_sha256"]}, output_dir / "passed.json")
    return result


if __name__ == "__main__":
    unittest.main()
