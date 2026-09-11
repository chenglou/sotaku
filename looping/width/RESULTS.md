# Baseline Width Results

All six wider models completed exactly 20K updates, with final and selected checkpoints evaluated on all 25K puzzles. Retrieved September 6, 2026. Configurations, source hashes, and exact puzzle/depth sampling match the recorded protocol and paired width-128 controls. All evaluations used FP32 inference; none reported nonfinite values.

## Final Checkpoints

Percent of puzzles solved. Each average includes all three seeds, including collapsed models.

| Hidden width | Parameters | Mean @1024 | Mean @2048 | Mean @4096 | Stable finals |
|---|---:|---:|---:|---:|---:|
| 128, archived controls | 796,937 | 96.392 | 95.749 | 64.281 | 2/3 |
| 160 | 1,241,929 | 97.349 | 97.049 | 96.505 | 3/3 |
| 192 | 1,785,225 | 98.221 | 96.989 | 66.620 | 1/3 |

The criterion recorded before training requires at least 90% at 1024, at least 85% at 4096, and a drop of at most five percentage points. The width-192 model scoring 88.044% at 4096 fails because its drop exceeds ten points; that is not the same as the other model's collapse to 13.364%.

| Width | Seed | @16 | @128 | @512 | @1024 | @2048 | @4096 | Change vs paired control @1024 / @4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | 20260907 | 78.224 | 94.216 | 97.020 | 97.156 | 97.112 | 97.028 | reference |
| 128 | 20260908 | 77.936 | 93.356 | 94.456 | 94.312 | 93.896 | 2.856 | reference |
| 128 | 20260909 | 77.944 | 94.164 | 97.604 | 97.708 | 96.240 | 92.960 | reference |
| 160 | 20260907 | 79.724 | 95.632 | 98.360 | 98.632 | 98.688 | 98.644 | +1.476 / +1.616 |
| 160 | 20260908 | 79.964 | 95.660 | 97.748 | 97.440 | 96.608 | 95.196 | +3.128 / +92.340 |
| 160 | 20260909 | 79.644 | 95.016 | 96.072 | 95.976 | 95.852 | 95.676 | -1.732 / +2.716 |
| 192 | 20260907 | 80.916 | 96.308 | 98.104 | 98.160 | 95.312 | 13.364 | +1.004 / -83.664 |
| 192 | 20260908 | 80.008 | 95.764 | 98.076 | 98.060 | 97.124 | 88.044 | +3.748 / +85.188 |
| 192 | 20260909 | 80.024 | 95.772 | 98.172 | 98.444 | 98.532 | 98.452 | +0.736 / +5.492 |

Both widths pass the predeclared accuracy screen: mean 1024 improves by at least 0.5 points, mean 4096 does not decline, and at least two paired 1024 scores improve. Width 160 also passes the stability screen. These aggregate criteria do not guarantee stable individual runs: width 192's mean hides one severe failure. Width 160's large mean 4096 improvement is mostly the absence of the control's one collapse. The subsequent [50K confirmation](../width_50k/RESULTS.md) also produced three stable finals.

## Selected Checkpoints

Selection used the earliest maximum 1024 score on the fixed 1K monitoring sample, not these full-set scores. Final checkpoints remain the primary comparison.

| Width | Seed | Selected update | @16 | @128 | @512 | @1024 | @2048 | @4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 160 | 20260907 | 19K | 79.532 | 95.552 | 98.364 | 98.700 | 98.668 | 98.612 |
| 160 | 20260908 | 19K | 79.720 | 95.584 | 97.740 | 97.120 | 95.692 | 93.944 |
| 160 | 20260909 | 18K | 78.944 | 94.864 | 96.124 | 96.080 | 95.964 | 95.444 |
| 192 | 20260907 | 19K | 80.760 | 96.212 | 98.152 | 98.216 | 98.216 | 74.380 |
| 192 | 20260908 | 19K | 79.984 | 95.752 | 98.084 | 98.044 | 96.004 | 82.912 |
| 192 | 20260909 | 17K | 79.360 | 95.560 | 98.080 | 98.368 | 98.392 | 98.252 |

Selecting by 1024 monitoring accuracy did not remove width 192's deep-iteration failures.

## Training Behavior And Cost

Late monitoring is the mean/minimum 1024 score across nine 1K-puzzle checks from updates 12K through 20K, inclusive. Times are minutes on one H200 per run. Training time excludes compilation, monitoring, checkpoint saving, and full evaluation.

| Width | Seed | Late mean / minimum | Training | Compilation | Full eval final / selected |
|---|---|---:|---:|---:|---:|
| 128 | 20260907 | 93.18 / 83.0 | 122.7 | 8.8 | 17.2 / 17.2 |
| 128 | 20260908 | 89.22 / 76.1 | 121.4 | 33.4 | 17.0 / 17.0 |
| 128 | 20260909 | 92.20 / 85.2 | 123.9 | 13.0 | 20.5 / 20.4 |
| 160 | 20260907 | 97.41 / 94.4 | 177.6 | 8.6 | 22.2 / 22.2 |
| 160 | 20260908 | 95.57 / 93.1 | 174.1 | 9.0 | 22.0 / 22.0 |
| 160 | 20260909 | 93.37 / 89.8 | 176.1 | 27.9 | 22.0 / 21.9 |
| 192 | 20260907 | 95.96 / 92.2 | 183.4 | 8.3 | 24.7 / 24.7 |
| 192 | 20260908 | 96.03 / 92.1 | 183.4 | 28.6 | 24.7 / 24.7 |
| 192 | 20260909 | 96.30 / 91.2 | 185.3 | 32.9 | 24.8 / 24.8 |

Average optimizer-update time was 122.7 minutes at width 128, 175.9 at width 160, and 184.0 at width 192, about 1.43x and 1.50x the control. Parameter counts grow by 1.56x and 2.24x. Width 160's seed 20260907 recovered from 28.3% at 1024 after 10K updates to 98.632% on the final full evaluation. Wider states did not eliminate mid-training regressions.

These results support testing modestly increased width, but not the claim that inadequate state size alone causes collapse. Widening changes both state size and model capacity. The benchmark has been reused during development; it is not an untouched holdout.

## Artifacts

The [compact records](results/completed_runs.json) preserve all six runs, 1024 learning curves, checkpoint checksums, timings, and both full evaluation profiles. Full artifacts remain on `sudoku-outputs` under `baseline_width_v1_20260905/runs/<arm>_seed<seed>/`, with local summaries in `runs_modal/baseline_width_v1_20260905/runs/`. Frozen control scores and identities are in [reference.json](reference.json). No default changes were made.
