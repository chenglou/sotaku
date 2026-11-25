"""Train the released recipe on a small part of the training split, e.g. 1,000 puzzles.

This is the setting of HRM and TRM: a random subset of the Sudoku-Extreme training puzzles,
each kept together with a fixed number of random symmetric copies. The trainer in
`stabilize/exp_testbed_20k.py` is pinned by checksum in archived studies, so it is not edited.
`train_small` instead swaps the training split that the trainer loads for the small one and
leaves the training loop, the test split and the monitoring puzzles unchanged.
"""

import contextlib
import functools
import os

import numpy as np

USUAL_TRAINING_POOL = 2_700_000  # the trainer uses the first 2.7M training puzzles


def random_sudoku_symmetry(rng):
    """One random transformation that maps valid Sudoku boards to valid Sudoku boards.

    Returns `(digit_map, source_cells)`: the new board has `digit_map[old[source_cells[i]]]`
    in cell `i`. The transformation relabels the digits, permutes the three bands and the rows
    inside each band, does the same for columns, and transposes the board half of the time.
    These are the augmentations of the Sudoku-Extreme dataset builder used by HRM and TRM.
    """
    digit_map = np.concatenate(([0], rng.permutation(np.arange(1, 10)))).astype(np.uint8)
    rows = np.concatenate([band * 3 + rng.permutation(3) for band in rng.permutation(3)])
    columns = np.concatenate([stack * 3 + rng.permutation(3) for stack in rng.permutation(3)])
    source_cells = rows[:, None] * 9 + columns[None, :]
    if rng.random() < 0.5:
        source_cells = source_cells.T
    return digit_map, source_cells.reshape(81)


def augmented_training_pool(question_digits, answer_digits, copies, rng):
    """Every board followed by `copies` random symmetric versions of it.

    `question_digits` and `answer_digits` are `(puzzles, 81)` arrays of digits 0-9, where 0 is
    an empty cell. Returns the two enlarged arrays and, for every row, the index of the puzzle
    it was made from.
    """
    question_digits = np.asarray(question_digits, dtype=np.uint8)
    answer_digits = np.asarray(answer_digits, dtype=np.uint8)
    if question_digits.shape != answer_digits.shape or question_digits.shape[1:] != (81,):
        raise ValueError("questions and answers must both have shape (puzzles, 81)")
    if not isinstance(copies, int) or copies < 0:
        raise ValueError("copies must be a non-negative integer")
    puzzles = question_digits.shape[0]
    per_puzzle = 1 + copies
    questions = np.empty((puzzles * per_puzzle, 81), dtype=np.uint8)
    answers = np.empty_like(questions)
    for puzzle in range(puzzles):
        start = puzzle * per_puzzle
        questions[start] = question_digits[puzzle]
        answers[start] = answer_digits[puzzle]
        for copy in range(1, per_puzzle):
            digit_map, source_cells = random_sudoku_symmetry(rng)
            questions[start + copy] = digit_map[question_digits[puzzle, source_cells]]
            answers[start + copy] = digit_map[answer_digits[puzzle, source_cells]]
    return questions, answers, np.repeat(np.arange(puzzles), per_puzzle)


def board_digits(boards):
    """Board strings such as "53..7...." as a `(boards, 81)` array of digits, 0 for empty."""
    codes = np.frombuffer("".join(boards).encode("ascii"), dtype=np.uint8).reshape(len(boards), 81)
    return np.where(codes == ord("."), 0, codes - ord("0")).astype(np.uint8)


def board_strings(digits):
    characters = np.where(digits == 0, ord("."), digits + ord("0")).astype(np.uint8)
    return [row.tobytes().decode("ascii") for row in characters]


class SmallTrainingSet:
    """Stands in for the training split: supports `len()` and `[:count]`, as the trainer needs."""

    def __init__(self, train_split, puzzles, augmentations, subset_seed):
        # The subset has its own random generator, so every run with the same `subset_seed`
        # trains on the same puzzles and the same augmented copies, whatever its training
        # seed is and however often it is resumed.
        rng = np.random.default_rng(subset_seed)
        pool = min(USUAL_TRAINING_POOL, len(train_split))
        self.source_rows = np.sort(rng.choice(pool, size=puzzles, replace=False))
        rows = train_split.select(self.source_rows.tolist())
        source_ratings = np.asarray(rows["rating"], dtype=np.int64)
        questions, answers, source_puzzle = augmented_training_pool(
            board_digits(rows["question"]), board_digits(rows["answer"]), augmentations, rng,
        )
        if answers.min() < 1:
            raise ValueError("a training answer has an empty cell")
        self.columns = {
            "question": board_strings(questions),
            "answer": board_strings(answers),
            "rating": source_ratings[source_puzzle].tolist(),
        }

    def __len__(self):
        return len(self.columns["rating"])

    def __getitem__(self, rows):
        if not isinstance(rows, slice):
            raise TypeError("the small training set only supports slices")
        return {name: values[rows] for name, values in self.columns.items()}


@contextlib.contextmanager
def small_training_split(trainer, puzzles, augmentations, subset_seed):
    """While active, `trainer.load_dataset(..., split="train")` returns the small set."""
    original = trainer.load_dataset
    built = {}

    def load_dataset(*args, **kwargs):
        dataset = original(*args, **kwargs)
        if kwargs.get("split") != "train":
            return dataset
        built["set"] = SmallTrainingSet(dataset, puzzles, augmentations, subset_seed)
        print(
            f"Small training set: {puzzles} puzzles (subset seed {subset_seed}), "
            f"{augmentations} augmented copies each, {len(built['set'])} boards in total"
        )
        return built["set"]

    trainer.load_dataset = load_dataset
    try:
        yield built
    finally:
        trainer.load_dataset = original


@contextlib.contextmanager
def adamw_weight_decay(trainer, weight_decay):
    """While active, the trainer's AdamW uses `weight_decay` (the recipe leaves it at 0.01)."""
    if weight_decay is None:
        yield
        return
    optimizers = trainer.torch.optim
    original = optimizers.AdamW
    optimizers.AdamW = functools.partial(original, weight_decay=weight_decay)
    try:
        yield
    finally:
        optimizers.AdamW = original


@contextlib.contextmanager
def peak_learning_rate(trainer, learning_rate):
    """While active, the trainer's schedule peaks at `learning_rate` (the recipe uses 2e-3).

    The checkpoint's saved config keeps the recipe's value, so the run's
    `small_data_<run>.json` file is the record of the learning rate that was used.
    """
    if learning_rate is None:
        yield
        return
    original = trainer.lr
    trainer.lr = learning_rate
    try:
        yield
    finally:
        trainer.lr = original


def train_small(output_dir, *, arm="late_state_ce", run_name, random_seed, puzzles=1000,
                augmentations=1000, subset_seed=0, weight_decay=None, stop_after_step=None,
                mixed_difficulty=False, learning_rate=None, total_steps=None):
    """The 50K-step recipe of `looping.exp_health_methods.train` on a small training set.

    `mixed_difficulty` samples all ratings from the first step, in place of the recipe's
    hard-to-easy phases. With 1,000 puzzles the first phase would otherwise train on the
    roughly 80 hardest ones alone for 10,000 steps.

    `total_steps` shortens the whole schedule, e.g. 10,000 steps with the cosine decay ending
    at step 10,000. It needs `mixed_difficulty`, because the phases have fixed step ranges.
    """
    import stabilize.exp_testbed_20k as trainer
    from checkpoint_utils import atomic_json_save
    from looping.exp_health_methods import get_health_method_config
    from looping.exp_stay_solved import FULL_50K_SCHEDULE

    settings = get_health_method_config(arm)
    if stop_after_step is not None:
        settings["stop_after_step"] = stop_after_step
    schedule = dict(FULL_50K_SCHEDULE)
    if total_steps is not None:
        if not mixed_difficulty:
            raise ValueError("total_steps needs mixed_difficulty")
        if total_steps <= schedule["warmup_steps"]:
            raise ValueError("total_steps must be longer than the warmup")
        schedule["total_steps"] = total_steps
    if mixed_difficulty:
        schedule["phases"] = ((0, schedule["total_steps"], 0, "All ratings from the start"),)
    os.makedirs(output_dir, exist_ok=True)
    record = {
        "run_name": run_name, "arm": arm, "random_seed": random_seed, "puzzles": puzzles,
        "augmentations": augmentations, "subset_seed": subset_seed,
        "weight_decay": 0.01 if weight_decay is None else weight_decay,
        "mixed_difficulty": mixed_difficulty,
        "learning_rate": trainer.lr if learning_rate is None else learning_rate,
        "total_steps": schedule["total_steps"],
    }
    with small_training_split(trainer, puzzles, augmentations, subset_seed) as built, \
            adamw_weight_decay(trainer, weight_decay), \
            peak_learning_rate(trainer, learning_rate):
        result = trainer.train(
            output_dir=output_dir, experiment_name=f"exp_health_{arm}", run_name=run_name,
            random_seed=random_seed, checkpoint_on_probe=True, schedule=schedule,
            **settings,
        )
    record["source_rows"] = built["set"].source_rows.tolist()
    atomic_json_save(record, os.path.join(output_dir, f"small_data_{run_name}.json"))
    return result

