# Study Status

Research branch: `codex/weight-tying-study`. Protocol recorded in commit `29c276d` before training or test-set evaluation. The v2 release and public defaults are unchanged.

Outputs: `sudoku-outputs/weight_tying_v1_20260902/`.

## Preparation

| Job | Modal App | Status |
|---|---|---|
| Generate and screen fresh puzzles | `ap-VTOGKyY7Ju9uzzTgLcSKZT` | Running |
| Initial GPU preflight | `ap-DVASDpLJBCJW0HRmMOuzpD` | Stopped before training; corrected the test's expectation of encoder gradients on gradient-free batches |
| Corrected full-batch GPU preflight | `ap-ZI82BAMLnEkyx1xTn7Rm68` | Running |

No study training or held-out evaluation has started. Training requires successful data preparation and a preflight whose source hashes match the worker's code. The extra gradient test confirms that the input encoder has no gradient on late-state batches while all subsequently used weights do.
