# Width 160: 50K Results

All three fresh runs completed exactly 50K updates and both full 25K-puzzle evaluations. Results retrieved September 6, 2026. Configurations, all 57 source hashes, data identities, and exact paired puzzle/depth sampling were verified against the preflight and archived controls. All evaluations used FP32 inference; no nonfinite results were reported.

## Final Checkpoints

Percent of puzzles solved. Final models are the primary comparison, with every seed included.

| Width | Seed | @16 | @128 | @512 | @1024 | @2048 | @4096 |
|---|---|---:|---:|---:|---:|---:|---:|
| 128, control | 20260910 | 81.076 | 96.060 | 98.280 | 98.244 | 97.720 | 91.688 |
| 128, control | 20260911 | 81.180 | 96.276 | 98.224 | 97.848 | 97.432 | 97.236 |
| 128, control | 20260912 | 80.764 | 96.432 | 98.884 | 99.228 | 99.368 | 99.436 |
| 160 | 20260910 | 82.192 | 96.956 | 99.012 | 99.188 | 99.084 | 98.584 |
| 160 | 20260911 | 82.260 | 96.904 | 99.068 | 99.144 | 99.028 | 98.784 |
| 160 | 20260912 | 82.304 | 96.588 | 98.644 | 98.732 | 98.648 | 98.500 |

| Width | Mean @1024 | Mean @2048 | Mean @4096 | Stable finals |
|---|---:|---:|---:|---:|
| 128 | 98.440 | 98.173 | 96.120 | 2/3 |
| 160 | 99.021 | 98.920 | 98.623 | 3/3 |
| Difference, percentage points | +0.581 | +0.747 | +2.503 | |

Paired 1024 changes were +0.944, +1.296, and -0.496 points; paired 4096 changes were +6.896, +1.548, and -0.936. The wider model improved two pairs, not every run. The control's unsuccessful seed missed the five-point maximum-drop criterion, rather than catastrophically collapsing.

**The predeclared confirmation passed:** mean 1024 improved by at least 0.5 points without reducing mean 4096, two 1024 pairs improved, and all three wider finals met the stability criterion. Each scored at least 98.5% at 4096, with drops of only 0.604, 0.360, and 0.232 points from 1024. The criterion itself required at least 90% at 1024, at least 85% at 4096, and a drop of at most five points.

This supports width 160 as a more reliable candidate at additional cost, not a guarantee. The earlier 20K screen also had three stable width-160 finals, on different seeds. The schedules differ, so those six observations are not six repeats of one identical recipe.

## Selected Checkpoints

Selection used the earliest maximum 1024 score on the fixed 1K monitoring sample, not the full-set results below.

| Seed | Selected update | @16 | @128 | @512 | @1024 | @2048 | @4096 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260910 | 50K | 82.192 | 96.956 | 99.012 | 99.188 | 99.084 | 98.584 |
| 20260911 | 39K | 80.676 | 96.364 | 98.848 | 99.120 | 99.028 | 98.396 |
| 20260912 | 47K | 82.324 | 96.692 | 98.780 | 98.800 | 98.712 | 98.576 |

Selection was unnecessary to avoid final-model collapse in these runs. The final checkpoint outscored the selected checkpoint in the second run, while the third run's selected checkpoint was slightly better.

## Training Behavior And Cost

Late monitoring is the mean/minimum 1024 accuracy across 21 checks of the fixed 1K sample from updates 30K through 50K, inclusive. Times are minutes on one H200 per run.

| Seed | Late mean / minimum | Optimizer updates | Compilation | Monitoring | Checkpoint saving | Full eval final / selected |
|---|---:|---:|---:|---:|---:|---:|
| 20260910 | 98.25 / 96.6 | 440.4 | 8.0 | 11.4 | 1.8 | 21.9 / 21.9 |
| 20260911 | 97.83 / 93.5 | 438.8 | 8.5 | 11.4 | 2.5 | 22.0 / 22.1 |
| 20260912 | 97.09 / 92.8 | 438.2 | 8.2 | 11.5 | 1.7 | 22.1 / 22.1 |

Average training time was 7h19m for optimizer updates, or 7h41m including compilation, monitoring, and checkpoint saving. Both full evaluations added about 44 minutes, for roughly 8h25m end to end. All three ran in parallel. Width 160 has 1,241,929 parameters versus 796,937 for the control, about 1.56x; optimizer-update time was about 1.44x the matched controls.

## Artifacts

The [compact records](results/completed_runs.json) preserve all three runs, 1024 learning curves, checkpoint checksums, timings, and both full evaluation profiles. Full artifacts remain on `sudoku-outputs` under `baseline_width_50k_v1_20260906/runs/width160_seed<seed>/`, with local summaries in `runs_modal/baseline_width_50k_v1_20260906/runs/`. All three controls and their identities remain in [reference.json](reference.json). The benchmark was reused during development and is not an untouched holdout.
