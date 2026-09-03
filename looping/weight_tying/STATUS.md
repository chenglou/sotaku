# Study Status

Research branch: `codex/weight-tying-study`. Protocol recorded in commit `29c276d` before training or test-set evaluation. The v2 release and public defaults are unchanged.

Outputs: `sudoku-outputs/weight_tying_v1_20260902/`.

## Preparation

| Job | Modal App | Status |
|---|---|---|
| Generate and screen fresh puzzles | `ap-VTOGKyY7Ju9uzzTgLcSKZT` | Complete; manifest committed to the Volume |
| Initial GPU preflight | `ap-DVASDpLJBCJW0HRmMOuzpD` | Stopped before training; corrected the test's expectation of encoder gradients on gradient-free batches |
| Corrected full-batch GPU preflight | `ap-ZI82BAMLnEkyx1xTn7Rm68` | Passed for all three architectures |

Data preparation finished at 2026-09-03 01:08 UTC after 870 seconds. The frozen set contains 10,000 puzzles, 2,500 per generator difficulty. All generated puzzles have one solution. Screening covered 3,831,994 original training rows and 422,786 test rows; no duplicate candidate questions or solved grids were found under the documented checks. The held-out NPZ checksum is `c8f610dffc04150525eb579c5fd8f98b7baf6d6d6185ff6c1a23dac6653b1cde`.

All 156 local tests pass, including exact interrupted/resumed training, weight-copy independence, gradient relationships, cohort locking, independent score recomputation, and read-only result collection. The full-batch GPU preflight passed early training and 32/512-iteration gradient-free prefixes for all three architectures, with checkpoint reloads. Peak allocated memory was 50.1 GiB for tied, 47.4 GiB for compute-matched untied, and 12.3 GiB for parameter-matched untied. Compilation and checks took about 47 minutes total on one H200.

The committed preflight result has SHA256 `29c171eba0650e1783cc3acbefd9f722e79df519e1daefed42b31d3f384a0bb9`. All source hashes were verified against local code before launching. The input encoder has no gradient on late-state batches while all subsequently used weights do.

## Training

All 18 preregistered 20K runs were submitted in separate detached invocations between 18:45 and 18:49 PDT on September 2. At the initial status check, Modal had allocated 10 workers; eight inputs were waiting for GPUs. App and function-call IDs are recorded in `jobs.json`. Do not launch replacements merely because a worker has not been allocated yet.

The first tied and parameter-matched early-training runs have passed 1K updates. Their matching sample digests were checked at common steps, and the tied run's resumable checkpoint and selected weights were verified on the Volume. All 10 allocated workers' live configurations match their intended architecture, training regime, seed, 20K schedule, batch size, and data manifest. Waiting inputs cannot have their live logs verified until workers start.

`watch.py` collects results from existing function-call IDs without launching, replacing, or cancelling workers. One initial result-read request timed out; the collector now retries connection errors, and the workers were unaffected.

At 19:29 PDT, the first parameter-matched late-state worker was still compiling its supervised continuation. A live `py-spy` stack showed `train_run` calling PyTorch's AOT autograd partitioner, which was running NetworkX's `preflow_push` inside `minimum_cut`. No optimizer updates had started on that worker. The profiler was installed only in `/tmp/sotaku-diagnostics`; training code and dependencies were unchanged. This is compilation overhead, not a measured training failure. Other workers were already saving checkpoints and advancing.

No held-out evaluation has started. After every run finishes, verify results and matching sample digests, seal the cohort, then evaluate both final and best-validation checkpoints using the commands in `README.md`. Final-checkpoint performance remains primary.
