# Frozen QQWing Test Set

`holdout.npz` contains the exact 10,000 puzzles used for the September 2026 weight-sharing study, with 2,500 puzzles per QQWing 1.3.4 difficulty category. It is a separate, easier distribution from sudoku-extreme, not a replacement benchmark for the v2 score.

SHA256: `c8f610dffc04150525eb579c5fd8f98b7baf6d6d6185ff6c1a23dac6653b1cde`.

Load with `np.load(path, allow_pickle=False)`. Arrays are `digits` (10,000 x 81, values 0-9 with 0 for empty cells), `targets` (10,000 x 81, classes 0-8 for digits 1-9), and `labels` (QQWing difficulty strings). Boards are flattened in row-major order.

Each puzzle has exactly one solution. Questions and completed grids were screened against all 3,831,994 training and 422,786 test rows in the pinned sudoku-extreme revision, including digit relabelings and the eight rotations/reflections. This does not establish non-equivalence under all Sudoku symmetries. Generator metadata and checksums are in the [data manifest](../results/data_manifest.json); construction code is in [data.py](../data.py). The generator and uniqueness solver are never used at neural inference time.

All 18 runs' final and validation-selected checkpoints were locked before evaluating this set. Its scores are now known, so future tuning cannot treat these same puzzles as an untouched test set. QQWing's CLI does not expose a reproducible seed; use these frozen bytes to repeat the study's evaluations.
