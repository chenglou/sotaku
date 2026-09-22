# Nearby-State Diagnostics

All six width-128 checkpoints completed the fixed protocol on September 9, 2026: three 20K models and three 50K models, 50 analysis puzzles per model, and 96 two-dimensional perturbation maps across two separate illustration puzzles. Each analysis took 3.6-4.1 minutes on an H200, excluding startup. No models were trained or modified, and no perturbed answer was selected for inference.

Start with the [accuracy curves](results/comparison.png) and [plot gallery](results/GALLERY.md). The [protocol](README.md) describes sampling, controls, and limitations; [summary.json](results/summary.json) contains the counts. The original full-benchmark scores remain in the [20K reference](../width/reference.json) and [50K reference](../width_50k/reference.json). The small sample below is not a replacement for those scores.

## What We Found

**The failing model usually finds the answer, then loses it.** On the 50 analysis puzzles, 20K seed 20260908 gets 48 correct at some point but ends with only two correct. Twenty-four puzzles finish on wrong answers that have not changed for at least 128 updates. Answer stability alone would therefore be a misleading success criterion.

**Small nudges do not broadly rescue this collapse.** With 0.1% perturbations at iteration 128, none of the 200 perturbed continuations repairs a final failure, and three lose an answer their paired unperturbed continuation retains. The largest late perturbations produce two rescues and four harms. These are repeated trials on the same 50 puzzles, not 200 independent puzzles or evidence about a training success rate.

**Some successful models recover much better than others.** At iteration 1024, the larger perturbations immediately damage the displayed answer on all 200 trials for both 20K seed 20260907 and 50K seed 20260912. By iteration 4096, the former recovers on 57 trials; the latter recovers on all 200. This comparison describes these checkpoints, not a controlled causal effect of training duration.

| Training Steps / Seed | Unperturbed Final Accuracy | Final Accuracy After 3% Nudge at 1024 | Immediately Damaged Trials That Recover |
|---|---:|---:|---:|
| 20K / 20260907 | 100% | 28.5% | 57 / 200 |
| 20K / 20260908 | 4% | 3% | 6 / 188 |
| 20K / 20260909 | 94% | 92.5% | 184 / 195 |
| 50K / 20260910 | 98% | 96% | 108 / 111 |
| 50K / 20260911 | 100% | 100% | 107 / 107 |
| 50K / 20260912 | 100% | 100% | 200 / 200 |

Each perturbed accuracy averages four signed-direction trials per puzzle. No best-of-four answer selection is used. Smaller 0.1% nudges at iteration 1024 do not change final correctness in any of the six models on this sample.

## Reading the Maps

The paper's two-dimensional state slices are useful diagnostics here, but the initial maps do not establish fractal structure.

The margin plot shows the smallest gap between a correct digit's score and its strongest wrong alternative, across the puzzle's empty cells. Positive means every empty cell favors the correct digit; negative means at least one favors a wrong digit.

- [Failing model, small nudge at 128](results/models/20k_20260908/map_row857_t128_r0.001_p2701.png): all 289 starts for puzzle 291480 begin correct and end on the same wrong board. Their last answer changes occur between iterations 3548 and 3555. The unperturbed trajectory's weakest correct-digit margin declines from about +10.45 to -3.09; sampled measurements first show it below zero at iteration 2592. Neighboring normalized states separate by at most 2.31 times their initial distance. This example is consistent with nearby trajectories drifting into the same failure, without needing large sensitivity to the starting nudge.
- [Healthy 20K model, larger nudge at 1024](results/models/20k_20260907/map_row857_t1024_r0.03_p2701.png): a roughly rounded central region keeps a correct final answer; many farther starts fail. Both independently chosen planes show this contrast. That is a finite region of robustness in these slices, not proof of a basin boundary's fractal dimension.
- [Healthy 50K model, same puzzle and larger nudge](results/models/50k_20260912/map_row857_t1024_r0.03_p2701.png): all 289 starts end correct. All sixteen maps for this checkpoint end correct at every grid point, although some answers change along the way.

The PCA panels show the motion of three fixed neighboring trajectories. Smooth, nearly overlapping curves can accompany either success or failure. These projections retain the dominant motion, not necessarily the small components that decide a digit. They are not evidence of a hidden fixed point, and separately fitted PCA axes cannot be compared directly across models.

## Important Scale Caveat

The nudge sizes are percentages of the current hidden-state RMS. For healthy 20K seed 20260907 on puzzle 291480, state RMS is 57.09 at iteration 128 and 513.70 at iteration 1024. A 3% per-axis nudge therefore grows from RMS 1.71 to 15.41. The later experiment applies roughly nine times as much absolute disturbance. We cannot attribute the larger late damage solely to an intrinsically less robust later state.

Matched absolute perturbations at both iterations remain untested. Separate follow-ups now include [finer recursive grids](recursive/deep/README.md) and [full FP64 zooms on slow-solving puzzles](slow_puzzles/RESULTS.md). Those maps show fine structure but do not establish a fractal dimension or justify a new training objective.

## Verification and Artifacts

The real GPU preflight matches the original model's recurrence exactly. All 15 diagnostic tests pass in the GPU image; the local core suite passes all 225 tests, and all 19 existing research test directories pass. The six live configurations were checked against the fixed protocol and checkpoint identities. All 96 maps have exactly matching repeated zero-nudge controls and no nonfinite trajectories. Downloaded results and plots pass their saved checksums; raw map arrays remain on the Modal volume. Raw files used for the numerical examples above were separately downloaded and checksum-checked.

The output root is `basin_diagnostics_v2_20260909/` on `sudoku-outputs`; detached app and call IDs are in [jobs.json](jobs.json). Version 1 contains only the initial smoke test. Use `python -m looping.basin_diagnostics.collect` to retrieve committed summaries, probe arrays, and figures, followed by `python -m looping.basin_diagnostics.report` to verify and regenerate the gallery. Add `--include-map-arrays` to the collector for the larger raw map files. The release checkpoint and inference settings are unchanged.
