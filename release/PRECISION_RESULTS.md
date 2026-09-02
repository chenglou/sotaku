# Numerical Precision Results

Checked 2026-09-02. These comparisons change execution settings, not weights, training, or puzzle selection. The [protocol](PRECISION_PROTOCOL.md) defines the fixed samples and the full-set follow-up. All runs use PyTorch 2.10.0+cu128 on H200, without damping or early stopping.

## Full 25K Benchmark

Batch size 256, ordinary recurrence. FP32 uses `matmul_precision=highest`; BF16 uses CUDA autocast and the historical `high` setting. Compiled runs use 16-iteration chunks.

| Weights | Execution | 128 | 1024 | 2048 | 4096 |
|---|---|---:|---:|---:|---:|
| Late-state CE | FP32, eager | 96.288% | 99.116% | 99.048% | 98.632% |
| Late-state CE | BF16, eager | 96.472% | 98.796% | 92.976% | 43.704% |
| Late-state CE | BF16, compiled | 96.372% | 98.952% | 97.600% | 92.512% |
| Published v1 | FP32, eager | 95.704% | 98.888% | 99.028% | 99.020% |
| Published v1 | BF16, eager | 95.256% | 98.876% | 98.844% | 84.888% |
| Published v1 | BF16, compiled | 95.572% | 98.744% | 98.784% | 98.748% |

The selected late-state checkpoint's large 4096 decline largely disappears with FP32: 24,658 puzzles solved instead of 10,926. Compilation also helps BF16 at that depth, but does not reproduce FP32. The arithmetic matters much more at 4096 than at 1024 for these checkpoints.

FP32 is now the default in `solve.py`, `iters.eval_more_iters`, and `modal_eval.py`. Use `--precision bf16` for historical reproduction. Training remains unchanged: BF16 autocast, compiled execution, and dropout during both burn-in and supervised training unless an experiment explicitly overrides the burn-in setting.

These are two selected checkpoints, not a retest of every earlier failed training seed. The result does not prove that FP32 repairs all collapses or that trajectories remain correct indefinitely. V1 still has the higher FP32 score at 4096, despite the late-state model's higher 1024 score.

## Fixed 1K Matrix

The sample is the first 200 puzzles per bucket from the frozen 25K, selected before model outputs were inspected. Its row-content checksum is `a9fc44c2c39b39e862e91396c99cd0ad131cd70a94eb47e0d36950d768623f84`. Raw results retain paired puzzle disagreements, not just total scores.

| Weights | Precision | Execution | Batch | 128 | 1024 | 2048 | 4096 |
|---|---|---|---:|---:|---:|---:|---:|
| Late-state CE | BF16 | eager | 256 | 96.1% | 98.1% | 93.3% | 42.2% |
| Late-state CE | BF16 | eager | 32 | 95.7% | 98.5% | 93.3% | 43.3% |
| Late-state CE | BF16 | compiled | 256 | 96.3% | 98.7% | 97.2% | 93.4% |
| Late-state CE | BF16 | compiled | 32 | 95.7% | 98.3% | 97.2% | 93.0% |
| Late-state CE | FP32 | eager | 256 | 95.5% | 99.0% | 98.7% | 97.9% |
| Late-state CE | FP32 | eager | 32 | 95.5% | 99.0% | 98.7% | 97.9% |
| Late-state CE | FP32 | compiled | 256 | 95.1% | 99.0% | 98.8% | 98.0% |
| Late-state CE | FP32 | compiled | 32 | 95.7% | 98.9% | 98.6% | 97.8% |
| Published v1 | BF16 | eager | 256 | 95.0% | 98.7% | 98.6% | 85.2% |
| Published v1 | BF16 | eager | 32 | 95.5% | 98.7% | 98.5% | 86.0% |
| Published v1 | BF16 | compiled | 256 | 95.5% | 98.5% | 98.5% | 98.4% |
| Published v1 | BF16 | compiled | 32 | 94.6% | 98.4% | 98.5% | 98.4% |
| Published v1 | FP32 | eager | 256 | 95.5% | 99.0% | 99.0% | 99.0% |
| Published v1 | FP32 | eager | 32 | 95.5% | 99.0% | 99.0% | 99.0% |
| Published v1 | FP32 | compiled | 256 | 94.8% | 98.6% | 98.8% | 98.8% |
| Published v1 | FP32 | compiled | 32 | 95.6% | 98.9% | 99.0% | 99.1% |

## Solution Retention

Observe every iteration through 4096 in eager execution at batch size 256. A regression means a board was correct and became incorrect on the next iteration. This is distinct from counting only failures at the final horizon. An additional check requires the observation pass to reproduce the original endpoint predictions exactly.

| Weights | Precision | Ever solved | Never lost after first solve | At least one regression |
|---|---|---:|---:|---:|
| Late-state CE | BF16 | 991 | 243 | 748 |
| Late-state CE | FP32 | 993 | 976 | 17 |
| Published v1 | BF16 | 988 | 825 | 163 |
| Published v1 | FP32 | 990 | 990 | 0 |

All four observation passes reproduced their corresponding eager endpoint predictions exactly, including the unscored given cells. FP32 sharply reduces regressions in this 1K sample, but the late-state model still loses some correct solutions. The full 25K tables above record endpoint gains and losses, not every intermediate event.

## Interpretation And Records

PyTorch documents that batching and arithmetic order can change floating-point results. Its Inductor 2.10 implementation also explains that fusion can retain FP32 intermediates where eager BF16 would round between operations. Those are plausible contributors, not an isolated explanation of which operation causes Sotaku's difference; changing precision can also change kernel selection. See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) and [the versioned Inductor configuration](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/config.py).

The recorded eager full reference runs took roughly 16.5 minutes in FP32 and 12-12.5 minutes in BF16. Cold compiled full runs varied much more, about 13.5 and 35 minutes including compilation; some 1K compiled conditions were slower still. These observations are not a controlled throughput comparison. Compilation is optional, not part of the recommended inference settings.

[Validation records](validation) retain model and dataset identities, exact indices, runtime packages and driver, source-file hashes, and per-puzzle predictions. Source archives preserve the precision-study and full-confirmation code versions. The original eager-BF16 full runs retain source hashes but not a separate complete source archive. Later metadata and CLI changes did not change their recurrence. `release_tools.verify_records` independently recomputes the endpoint scores from the pinned dataset and saved predictions. Large arrays, repeated index files, and source archives are included in the [prepared release archive](v2/README.md); they are excluded from Git.
