# Study Status

**Complete, September 3, 2026:** all 18 exact-20K training runs and all 36 full evaluations. No numerical failures or training-worker restarts. Do not launch replacement jobs. [Results and plots](RESULTS.md) give the conclusions; the v2 checkpoint and public defaults are unchanged.

Research branch: `codex/weight-tying-study`. Protocol committed in `29c276d` before training or test-set evaluation. Durable artifacts are on `sudoku-outputs` under `weight_tying_v1_20260902/`.

## Completion Record

| Stage | Record |
|---|---|
| Data preparation | `ap-VTOGKyY7Ju9uzzTgLcSKZT`; completed September 2, 18:08 PDT |
| Corrected full-batch H200 preflight | `ap-ZI82BAMLnEkyx1xTn7Rm68`; passed all three architectures |
| Training | [18 job IDs](jobs.json); launched September 2, 18:45-18:49 PDT; all results downloaded by 23:24 |
| Pre-evaluation audit | [Audit](pre_evaluation_audit.json); September 2, 23:24 PDT |
| Checkpoint selection lock | `ap-4KEwak9KkdaEoM15a1t0ZB`; [lock](cohort_lock.json) created September 2, 23:26 PDT |
| Full evaluation | [36 job IDs](evaluation_jobs.json); launched September 2, 23:27-23:41 PDT; all results retrieved by September 3, 01:01 |
| Download and verification | [Final artifact audit](results/artifact_audit.json); all saved predictions checked locally |

The new 10K QQWing set has 2,500 puzzles per difficulty. Screening covered all 3,831,994 sudoku-extreme training rows and 422,786 test rows under the checks described in the protocol. No candidate question or solution duplicates were found. The [frozen accepted puzzles](test_data/README.md) are included in the repository; generator output and uniqueness-check records remain on the Volume.

## Verification

- All 18 training results match the full registered configs, source hashes, exactly 20K updates, and paired puzzle/horizon digests. Same-width paired runs have identical nominal transformer matmul FLOPs.
- The best checkpoint is the earliest maximum of the registered validation metric. Both final and selected weights and manifests were verified and locked before opening the new set.
- All 36 evaluations match their selected weight hashes, locked data, and returned function-call results. Runtime records show H200, PyTorch 2.10.0+cu128, eager FP32, and TF32 matmul disabled.
- All 6.3 million saved board predictions were checked against their recorded exact-puzzle scores, difficulty counts, and endpoint gains/losses. Dataset and prediction checksums match.
- All 132 validation-score comparisons reconstructed from the full evaluations match the training records exactly. Six final/best export pairs share a training step; their separately evaluated prediction arrays are identical on both datasets.
- All 165 local tests pass, covering exact resumed training, independent weight storage and gradients, selection locking, score recomputation, rendering, result-collection transport errors, and validation replay. The GPU preflight covered full-batch early training and 32/512-iteration gradient-free prefixes, finite gradients, and checkpoint reloads.

Lock SHA256: `1fdc572cdbc32ddbec6162bee0a83d7881d53a46d46e2f34be61c2b994c366f0`. Preflight result SHA256: `29c171eba0650e1783cc3acbefd9f722e79df519e1daefed42b31d3f384a0bb9`.

## Operational Issues

The initial preflight (`ap-DVASDpLJBCJW0HRmMOuzpD`) stopped before training because a test incorrectly expected input-encoder gradients through a detached prefix. The test was corrected; the second preflight passed without changing model or training behavior. It took about 47 minutes for all three architectures.

Two narrow late-training runs spent unusually long compiling. A live stack sample showed PyTorch's AOT autograd partitioner inside NetworkX minimum-cut, not stalled optimizer updates. Both runs completed without intervention; compilation is reported separately from training.

The local result collector encountered Modal connection errors and an unwrapped `grpclib.exceptions.StreamTerminatedError`. It was updated to retry reads and restarted with its saved cache. Remote workers were unaffected. A replacement training run was never launched. This distinction is also documented in `tips-for-running-modal`.

Weights and resume checkpoints are under `runs/<run_name>/`; full predictions and environment records are under `evaluations/<run_name>/<selection>/`. Download into a new existing directory and verify hashes. The local 559 MB evaluation archive is not committed; the reports, histories, audit records, plots, and frozen new test puzzles are.
