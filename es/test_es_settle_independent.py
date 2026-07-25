import os
import tempfile
import unittest

import torch

from es.es_sampling import DEFAULT_SAMPLING_MODE, POPULATION_SIZE
from es.exp_es_settle_independent import CONFIG, save_best_probe
from es.modal_es_polish import resolve_run_paths


class IndependentSettlednessTest(unittest.TestCase):
    def test_recipe_uses_independent_sampling(self):
        self.assertEqual(DEFAULT_SAMPLING_MODE, "independent")
        self.assertEqual(CONFIG["sampling_mode"], "independent")
        self.assertEqual(
            CONFIG["population_evaluations"],
            POPULATION_SIZE,
        )
        self.assertEqual(CONFIG["fitness"], "solved_and_settled")
        self.assertEqual(CONFIG["fitness_iters"], 2048)

    def test_polish_paths_keep_source_and_outputs_separate(self):
        source, output, run_name = resolve_run_paths(
            "looping/model_candidate.pt",
        )
        self.assertEqual(
            source,
            "/outputs/looping/model_candidate.pt",
        )
        self.assertEqual(
            output,
            "/outputs/es_polish/model_candidate_settle_independent",
        )
        self.assertEqual(
            run_name,
            "model_candidate_settle_independent",
        )

    def test_polish_rejects_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "unsafe model path"):
            resolve_run_paths("../model.pt")

    def test_best_probe_weights_are_saved_and_not_overwritten_by_worse_probe(
        self,
    ):
        model = torch.nn.Linear(1, 1, bias=False)
        with tempfile.TemporaryDirectory() as output_dir:
            best_model_path = os.path.join(output_dir, "best.pt")
            with torch.no_grad():
                model.weight.fill_(1.0)
            best, saved = save_best_probe(
                model,
                {"generation": 0, "solved": 9, "settled": 9,
                 "both": 8, "total": 10},
                {"generation": -1, "both": -1},
                best_model_path,
            )
            self.assertTrue(saved)
            self.assertEqual(best["generation"], 0)

            with torch.no_grad():
                model.weight.fill_(2.0)
            best, saved = save_best_probe(
                model,
                {"generation": 5, "solved": 8, "settled": 9,
                 "both": 7, "total": 10},
                best,
                best_model_path,
            )
            self.assertFalse(saved)
            saved_state = torch.load(
                best_model_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(saved_state["weight"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
