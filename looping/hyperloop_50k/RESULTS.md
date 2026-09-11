# Hyperloop: 50K Results

All six runs completed exactly 50K updates and both full 25K-puzzle evaluations on September 5, 2026. Four gated states improved average accuracy at 1024 but did not pass the confirmation criteria because mean accuracy at 4096 declined and one final model collapsed.

## Final Checkpoints

Percent of puzzles solved on the frozen benchmark, using ordinary eager FP32 inference. Final checkpoints are primary; every seed is included.

| Variant | Seed | @1024 | @2048 | @4096 | Stable final |
|---|---|---:|---:|---:|---|
| Baseline | 20260910 | 98.244 | 97.720 | 91.688 | No |
| Baseline | 20260911 | 97.848 | 97.432 | 97.236 | Yes |
| Baseline | 20260912 | 99.228 | 99.368 | 99.436 | Yes |
| Four gated states | 20260910 | 99.260 | 99.320 | 99.268 | Yes |
| Four gated states | 20260911 | 99.380 | 96.688 | 29.328 | No |
| Four gated states | 20260912 | 99.116 | 99.188 | 99.176 | Yes |
| Baseline mean | | 98.440 | 98.173 | 96.120 | 2/3 |
| Four-state mean | | 99.252 | 98.399 | 75.924 | 2/3 |

The first two pairs improved at 1024; the third decreased by 0.112 percentage points. A stable final model required at least 90% at 1024, at least 85% at 4096, and a drop of at most five points. The unsuccessful baseline missed the drop criterion by 1.556 points; that is different from the four-state model's 70.052-point collapse. Better average 1024 accuracy did not establish better long-iteration reliability.

## Selected Checkpoints

Selection used the earliest best 1024 score on the fixed 1K monitoring sample.

| Variant | Seed | Selected update | @1024 | @2048 | @4096 |
|---|---|---:|---:|---:|---:|
| Baseline | 20260910 | 39K | 97.988 | 97.712 | 97.200 |
| Baseline | 20260911 | 47K | 97.844 | 97.324 | 97.044 |
| Baseline | 20260912 | 44K | 99.128 | 99.264 | 99.372 |
| Four gated states | 20260910 | 46K | 99.288 | 99.328 | 99.316 |
| Four gated states | 20260911 | 42K | 99.452 | 99.384 | 58.476 |
| Four gated states | 20260912 | 38K | 98.952 | 99.000 | 99.000 |

Selecting a checkpoint by its 1024 monitoring score did not remove the four-state failure at 4096. Full scores at 16, 128, and 512 are also retained in the [result records](results/completed_runs.json).

## Training And Cost

From updates 30K through 50K, inclusive, mean/minimum 1024 monitoring scores were 92.46/32.4, 95.45/79.7, and 98.28/95.5% for baseline; four-state scores were 98.52/94.6, 98.78/96.8, and 98.21/94.1%. Strong late-training behavior at 1024 did not prevent the later failure at 4096.

Average optimizer-update time was 305.6 minutes for baseline and 369.8 minutes for four states, about 21% more on one H200. Each full evaluation took about 17.1 or 19.5 minutes respectively; both final and selected checkpoints were evaluated. These timings exclude compilation, monitoring, and checkpoint saving. Parameters increased from 796,937 to 803,096, less than 1%.

## Records

The [compact records](results/completed_runs.json) preserve all six runs, 1024 learning curves, checkpoint and raw-result checksums, timings, and both full evaluation profiles. Cleanup verified saved configurations, all source hashes, paired initializations and sampling, earliest-best selection, and solved-count consistency. Individual prediction arrays were not re-scored during this cleanup. The benchmark was reused during development and is not an untouched holdout.

Full artifacts remain on `sudoku-outputs` under `hyperloop_50k_v1_20260905/runs/<arm>_seed<seed>/`; [jobs.json](jobs.json) contains the original call IDs. The [subsequent width experiments](../width/RESULTS.md) explored widening the ungated state instead of adding gated streams.
