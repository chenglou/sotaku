import types
import unittest

import numpy as np

from paper.small_data import (
    SmallTrainingSet, adamw_weight_decay, augmented_training_pool, board_digits, board_strings,
    peak_learning_rate, random_sudoku_symmetry, small_training_split,
)

QUESTION = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
ANSWER = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def is_valid_solution(board):
    grid = board.reshape(9, 9)
    full = set(range(1, 10))
    rows = all(set(grid[row]) == full for row in range(9))
    columns = all(set(grid[:, column]) == full for column in range(9))
    boxes = all(
        set(grid[row:row + 3, column:column + 3].reshape(-1)) == full
        for row in range(0, 9, 3) for column in range(0, 9, 3)
    )
    return rows and columns and boxes


class FakeSplit:
    """The parts of a Hugging Face dataset that `SmallTrainingSet` uses."""

    def __init__(self, count):
        rng = np.random.default_rng(3)
        questions, answers, _ = augmented_training_pool(
            board_digits([QUESTION]), board_digits([ANSWER]), count - 1, rng)
        self.columns = {"question": board_strings(questions), "answer": board_strings(answers),
                        "rating": list(range(count))}

    def __len__(self):
        return len(self.columns["rating"])

    def select(self, rows):
        selected = FakeSplit.__new__(FakeSplit)
        selected.columns = {name: [values[row] for row in rows]
                            for name, values in self.columns.items()}
        return selected

    def __getitem__(self, name):
        return self.columns[name]


class SmallDataTest(unittest.TestCase):
    def test_copies_are_valid_and_consistent(self):
        questions, answers, source = augmented_training_pool(
            board_digits([QUESTION]), board_digits([ANSWER]), 200, np.random.default_rng(0))
        self.assertEqual(questions.shape, (201, 81))
        self.assertTrue((source == 0).all())
        self.assertEqual(board_strings(questions[:1]), [QUESTION])
        self.assertEqual(board_strings(answers[:1]), [ANSWER])
        for question, answer in zip(questions, answers):
            self.assertTrue(is_valid_solution(answer))
            given = question > 0
            self.assertEqual(int(given.sum()), 81 - QUESTION.count("."))
            np.testing.assert_array_equal(question[given], answer[given])
        self.assertGreater(len({bytes(row) for row in questions}), 195)

    def test_symmetry_is_a_permutation(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            digit_map, source_cells = random_sudoku_symmetry(rng)
            self.assertEqual(digit_map[0], 0)
            self.assertEqual(sorted(digit_map[1:].tolist()), list(range(1, 10)))
            self.assertEqual(sorted(source_cells.tolist()), list(range(81)))

    def test_small_set_is_reproducible_and_keeps_ratings(self):
        split = FakeSplit(50)
        first = SmallTrainingSet(split, puzzles=5, augmentations=3, subset_seed=11)
        second = SmallTrainingSet(split, puzzles=5, augmentations=3, subset_seed=11)
        other = SmallTrainingSet(split, puzzles=5, augmentations=3, subset_seed=12)
        self.assertEqual(len(first), 20)
        self.assertEqual(first[:20], second[:20])
        self.assertNotEqual(first.source_rows.tolist(), other.source_rows.tolist())
        rows = first[:20]
        # Ratings in the fake split equal the row index, so each puzzle's four boards share one.
        self.assertEqual(rows["rating"], np.repeat(first.source_rows, 4).tolist())
        self.assertEqual(rows["question"][0], split.columns["question"][first.source_rows[0]])
        for question, answer in zip(rows["question"], rows["answer"]):
            self.assertTrue(is_valid_solution(board_digits([answer])[0]))
            self.assertTrue(all(q in (".", a) for q, a in zip(question, answer)))

    def test_only_the_training_split_is_replaced(self):
        split = FakeSplit(30)
        trainer = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: split)
        original = trainer.load_dataset
        with small_training_split(trainer, 4, 2, 0) as built:
            self.assertIs(trainer.load_dataset("name", split="test"), split)
            small = trainer.load_dataset("name", split="train")
            self.assertIs(small, built["set"])
            self.assertEqual(len(small), 12)
        self.assertIs(trainer.load_dataset, original)

    def test_weight_decay_is_restored(self):
        import torch

        trainer = types.SimpleNamespace(torch=torch)
        original = torch.optim.AdamW
        parameter = torch.nn.Parameter(torch.zeros(1))
        with adamw_weight_decay(trainer, 0.5):
            self.assertEqual(torch.optim.AdamW([parameter], lr=1.0).defaults["weight_decay"], 0.5)
        self.assertIs(torch.optim.AdamW, original)
        with adamw_weight_decay(trainer, None):
            self.assertIs(torch.optim.AdamW, original)

    def test_learning_rate_is_restored(self):
        trainer = types.SimpleNamespace(lr=2e-3)
        with peak_learning_rate(trainer, 5e-4):
            self.assertEqual(trainer.lr, 5e-4)
        self.assertEqual(trainer.lr, 2e-3)
        with peak_learning_rate(trainer, None):
            self.assertEqual(trainer.lr, 2e-3)


if __name__ == "__main__":
    unittest.main()
