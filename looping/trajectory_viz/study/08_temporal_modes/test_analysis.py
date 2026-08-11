import importlib
import unittest

import numpy as np


analysis = importlib.import_module(
    "looping.trajectory_viz.study.08_temporal_modes.analysis"
)
balanced_split_indices = analysis.balanced_split_indices
evaluate_dmd = analysis.evaluate_dmd
fit_dmd = analysis.fit_dmd
phase_randomize = analysis.phase_randomize
shuffle_time = analysis.shuffle_time
temporal_summary = analysis.temporal_summary


class TemporalModesTest(unittest.TestCase):
    def test_balanced_split_keeps_equal_bucket_counts(self):
        buckets = ["easy"] * 6 + ["hard"] * 6
        splits = balanced_split_indices(buckets)
        self.assertEqual(
            {name: len(indices) for name, indices in splits.items()},
            {"discovery": 4, "validation": 4, "final": 4},
        )
        for indices in splits.values():
            self.assertEqual(sum(index < 6 for index in indices), 2)
            self.assertEqual(sum(index >= 6 for index in indices), 2)

    def test_phase_randomization_preserves_channel_power(self):
        rng = np.random.default_rng(4)
        sequences = rng.normal(size=(3, 64, 5))
        randomized = phase_randomize(sequences, np.random.default_rng(5))
        original_power = np.abs(np.fft.rfft(sequences, axis=1))
        randomized_power = np.abs(np.fft.rfft(randomized, axis=1))
        np.testing.assert_allclose(randomized_power, original_power, atol=1e-10)

    def test_shuffle_time_preserves_values_but_changes_order(self):
        sequences = np.arange(40).reshape(2, 10, 2)
        shuffled = shuffle_time(sequences, np.random.default_rng(7))
        for puzzle in range(2):
            self.assertEqual(
                sorted(map(tuple, shuffled[puzzle])),
                sorted(map(tuple, sequences[puzzle])),
            )
        self.assertFalse(np.array_equal(shuffled, sequences))

    def test_dmd_recovers_rotation_and_beats_persistence(self):
        angle = 0.15
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        sequences = []
        for phase in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            values = [np.array([np.cos(phase), np.sin(phase)])]
            for _ in range(63):
                values.append(values[-1] @ rotation)
            sequences.append(values)
        sequences = np.asarray(sequences)
        matrix, bias = fit_dmd(sequences)
        metrics = evaluate_dmd(sequences, matrix, bias)
        self.assertGreater(metrics["r2"], 0.999)
        self.assertGreater(metrics["relative_to_persistence"], 0.999)

    def test_temporal_summary_separates_line_from_oscillation(self):
        time = np.arange(128)
        line = np.stack([time, 2 * time], axis=-1)[None].astype(float)
        oscillation = np.stack(
            [np.sin(2 * np.pi * time / 8), np.cos(2 * np.pi * time / 8)],
            axis=-1,
        )[None]
        line_summary = temporal_summary(line)
        oscillation_summary = temporal_summary(oscillation)
        self.assertGreater(line_summary["turn_cosine_mean"], 0.99)
        self.assertGreater(oscillation_summary["high_frequency_power_fraction"], 0.9)
        self.assertLess(abs(oscillation_summary["dominant_period"] - 8), 0.1)


if __name__ == "__main__":
    unittest.main()
