# Jacobian Precision Results

The original large spectral-radius estimates were numerically unreliable. FP64 automatic derivatives give radii around 1-2 at all 15 iteration-16 points, while the historical FP32-high method reports roughly 47-95 at the same operating states. The original claims about their magnitude and correlation with collapse are withdrawn.

All three runs completed through iteration 4096. Of 60 operating states, 38 produced validated radii and 22 remain unresolved because the eigenvalue solves did not meet the recorded convergence requirements. Runtime was 38.5 minutes for the original successful checkpoint, 15.4 minutes for the original collapsing checkpoint, and 30.2 minutes for v2, on separate H200s.

## Completed Comparison

The same five randomly selected puzzles are used for every checkpoint, one per difficulty bucket. Each reported radius passed two independently seeded ARPACK solves, eigenvector residual checks, automatic-derivative linearity, and FP64 central finite-difference checks. The [protocol](README.md) describes the selection and numerical settings. Medians and ranges below include only validated points; partial rows cannot be treated as complete five-puzzle comparisons. Solved counts include all five puzzles.

| Checkpoint | Iteration | Validated Radii | Radius, Median [Range] | Puzzles Solved |
|---|---:|---:|---:|---:|
| Original successful | 16 | 5/5 | 1.113910 [1.088958, 1.351220] | 3/5 |
| Original successful | 256 | 1/5 | 1.006848 | 5/5 |
| Original successful | 1024 | 1/5 | 1.002126 | 5/5 |
| Original successful | 4096 | 0/5 | Unresolved | 5/5 |
| Original collapsing | 16 | 5/5 | 1.093249 [1.072708, 1.964334] | 2/5 |
| Original collapsing | 256 | 5/5 | 1.191910 [1.065428, 1.627325] | 1/5 |
| Original collapsing | 1024 | 4/5 | 1.309239 [1.106160, 1.942153] | 0/5 |
| Original collapsing | 4096 | 3/5 | 1.328893 [1.064230, 1.928862] | 0/5 |
| Released Sotaku 2 | 16 | 5/5 | 1.084511 [1.055506, 1.356265] | 3/5 |
| Released Sotaku 2 | 256 | 5/5 | 1.022112 [1.008325, 1.032300] | 5/5 |
| Released Sotaku 2 | 1024 | 2/5 | 1.011976 [1.010722, 1.013231] | 5/5 |
| Released Sotaku 2 | 4096 | 2/5 | 1.007540 [1.004995, 1.010085] | 5/5 |

At iteration 16, the old-style FP32-high gain medians are 53.04, 70.16, and 59.72 for these three checkpoints, respectively. The original successful and v2 checkpoints retain correct answers through the measured later iterations despite validated radii above one. The failing checkpoint has larger validated later values, but incomplete solves and one checkpoint per condition prevent a general stability rule. These five puzzles do not estimate benchmark accuracy or training reliability.

## Precision Control

For the original successful model, puzzle 144160 at iteration 16:

| Method | Result |
|---|---:|
| FP64 automatic derivative, two independently seeded eigenvalue solves | 1.09707148 for both |
| FP64 historical finite-difference power iteration | 1.09565 |
| FP32 historical method, reduced-precision matmul disabled | 1.12873 |
| FP32 historical method, reduced-precision matmul enabled (`high`) | 52.53053 |

At the historical epsilon of 0.001, the relative error of random-direction forward differences is about 65-67 with FP32-high, about 0.35 with FP32-highest, and approximately 8e-8 with FP64. Relative error is the norm of the difference from the automatic derivative, divided by the automatic derivative's norm. Both FP64 solver seeds agree on the leading eigenvalue, with residuals below 1e-14.

Reduced-precision arithmetic dominates the historical estimate in this control. Disabling reduced-precision matmul helps, but FP32 finite differences remain inaccurate; changing that setting alone does not validate the estimator. The raw finite-difference outputs are retained as numerical controls and must not be relabeled as spectral radii.

The controls evaluate the same FP64 state, rounded to FP32 where needed, using math SDPA. They do not exactly replay the original study's FP32 trajectories, puzzle sample, or attention kernels, and they do not reconstruct every old table entry. A local Jacobian at a moving state cannot by itself establish convergence, divergence, or indefinite answer stability. A fixed point satisfies `F(h*) = h*` and does not require a zero Jacobian. This study does not estimate the largest singular value or prove contraction of the full map.

## Verification and Artifacts

Both local and GPU preflight suites passed all 10 tests. All three workers verified checkpoint hashes, matching puzzle IDs, FP64 activation dtypes, math SDPA, and hashes for all 16 recorded source files. The three completion manifests are present on the Volume; checksums for all 66 listed result artifacts match the downloads. The summarizer also checks that every completion record agrees with its individually saved point and that all 60 expected operating states are present.

[summary.json](summary.json) contains the aggregated values. [Job records](jobs.json) identify the completed runs, and [results/](results/) contains the preflight, environment records, and measurements, including unresolved eigenvalue solves and all precision controls. Raw numerical outputs are retained for reproducibility.
