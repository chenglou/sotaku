# Per-cell temporal geometry

## Bottom line

Sotaku has real low-dimensional and temporally ordered per-cell dynamics, but the results do not support one shared number helix. Raw hidden states are dominated by translation, while updates commonly trace one smooth arc and then backtrack. Time shuffles destroy much of the smooth ordering, but random three-dimensional planes usually recover almost the same phase linearity as the train-fitted PCA plane. The temporal structure is real; a privileged rotation plane is not.

The strongest isolated periodic result is the `late_state_ce` raw-state trajectory during iterations 64–128: a train-selected 0.75-cycle model beats translation by 0.601 [0.450, 0.695] and a four-parameter cubic by 0.551 [0.505, 0.597] on held-out suffixes. The same cells accumulate only 0.028 net turns after chord removal, have rotation monotonicity 0.072, and have lower phase linearity in the fitted plane than in random planes (0.284 versus 0.440). This is evidence for a local periodic component with cell-specific coefficients, not a common axis, phase, or repeated coil.

The spiral hypothesis is weaker. Spiral prediction loses to translation with a confidence interval below zero in 78 of 80 model × phase × representation comparisons. The only robust positive case is a 0.25-cycle fit to the stable model's deep unit-state trajectory, even though that trajectory has only 0.0095 total turns and 0.00027 net turns. A quarter-cycle smooth curve with almost no angular travel is not a spiral.

There is also no natural numeric `1→2→…→9→1` order. The exact test finds no significant natural cycle in any of 40 hidden-state endpoint tests, in the output-head contrast nullspace, in output-head rows, or in observed prediction changes. Model-specific nonnumeric cycles selected on fit puzzles do generalize strongly to held-out puzzles. That is a reproducible organization of digit identity, but it is static, model-specific, and generally unrelated to the sequence of digit changes.

## What was tested

The analysis uses 50 balanced evaluation puzzles: 10 from each rating bucket `0`, `1-2`, `3-10`, `11-50`, and `51+`. Alternating puzzles inside every bucket produce 25 fit puzzles and 25 held-out puzzles, with five puzzles from every bucket in each split. Feature bases and periodic frequencies are fitted only on the fit puzzles; trajectory scores, correctness events, and fitted digit orders are evaluated on the held-out puzzles. The output-head row analysis is a checkpoint-intrinsic control and does not use puzzles. Blank cells are primary; givens are a control.

Four checkpoints are compared:

- `stable_plain`: the stable plain baseline.
- `collapsed_plain`: the plain checkpoint that solves by iteration 128 and then degrades.
- `late_state_ce`: the checkpoint trained on later iterations.
- `combined_margin`: later-iteration training through step 39K, then a second supervised window and margin penalty through step 50K.

The run records every hidden state `h_t`, update `u_t = h_{t+1} - h_t`, and output distribution for iterations 0–1024. Geometry is summarized in five phases: 0–16, 16–64, 64–128, 128–512, and 512–1024. Long phases use at most 65 evenly spaced samples; correctness transitions use every iteration.

A helix-like claim should satisfy several conditions together:

1. Cells should accumulate meaningful net turns, rather than a large absolute turn count caused by reversals.
2. Unwrapped phase should progress monotonically with iteration.
3. A train-fitted plane should generalize to held-out puzzles and outperform arbitrary planes.
4. A periodic model should predict a held-out suffix better than translation, a same-parameter smooth curve, and shuffled-time controls.
5. A number helix should give the digits a consistent cyclic order, with the natural numeric order distinguished from the 20,159 alternatives.

For states, the rotation analysis first removes the chord between each phase's endpoints. For updates, it centers each cell's projected update path over time. A cell is included only when the projected radius is at least 10% of the representation-wide median and the plane contains at least 5% of that cell's three-dimensional energy. Rotation metrics include total and net turns, monotonicity `net / total`, phase-linearity R², radius variation, and radial-versus-angle R².

The predictive test is deliberately harder than an in-sample projection. It chooses a shared frequency from 0.125–3 cycles using fit puzzles, fits nuisance coefficients separately for each held-out cell on the first 60% of the phase, and scores the remaining 40%. Translation has columns `[1, t]`; rotation adds `[cos(ωt), sin(ωt)]` and is compared with a four-parameter cubic; spiral also adds `[t cos(ωt), t sin(ωt)]` and is compared with a six-parameter quintic. Full time shuffles, block shuffles, and reversal reuse the fit-selected frequency. Reversal is a backward-extrapolation check, not a negative control, because all of these model classes are reversible.

Most confidence intervals are 500-repeat rating-stratified puzzle bootstraps over the 25 held-out puzzles. The exact digit test enumerates all 20,160 unoriented cycles on nine labeled digits.

## Translation and low-dimensional structure

Train-puzzle PCA bases generalize. Across the four representations and four models, the fitted top three dimensions explain 0.309–0.618 of held-out energy, compared with a 95th percentile of at most 0.051 for random three-dimensional subspaces. This establishes compact per-cell motion, not a helix.

Raw hidden states become strongly translational. Deep-phase raw-state translation R² is 0.9998 for `stable_plain`, 0.566 for `collapsed_plain`, 0.867 for `late_state_ce`, and 0.983 for `combined_margin`. Unit normalization reduces the corresponding values to 0.804, 0.378, 0.504, and 0.815, so state-norm growth explains part, but not all, of the straight motion. The held-out suffix translation model has a positive 95% interval in 19 of 20 raw-state model × phase comparisons.

The same table also shows that the collapsed checkpoint's loss of straight deep motion coincides with its correctness collapse:

| model | deep raw-state translation R² | deep unit-state translation R² | t=1024 blank / puzzle accuracy | t=1024 margin median / p10 | deep gains / losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stable_plain` | 0.9998 | 0.804 | 1.000 / 1.000 | 796.3 / 493.2 | 0 / 0 |
| `collapsed_plain` | 0.566 | 0.378 | 0.755 / 0.040 | 51.7 / -2.21 | 324 / 647 |
| `late_state_ce` | 0.867 | 0.504 | 1.000 / 1.000 | 16.1 / 11.2 | 0 / 0 |
| `combined_margin` | 0.983 | 0.815 | 1.000 / 1.000 | 12.9 / 10.2 | 0 / 0 |

The state and update bases are not interchangeable. The mean squared principal cosine between raw three-dimensional state and update subspaces is 0.703, 0.548, 0.267, and 0.507 for the four models; the minimum principal cosine is 0.504, 0.172, 0.025, and 0.075. After unit normalization, the corresponding mean squared overlaps are 0.232, 0.532, 0.247, and 0.335. A curve seen in update coordinates should not be described as the same latent manifold as the state trajectory.

## Rotation, iteration phase, and controls

Detrended states do not accumulate repeated rotations. Across the 20 model × phase settings, raw-state residuals have 0.057–0.934 total turns but only 0.0015–0.290 net turns; the medians are 0.515 and 0.047. Median rotation monotonicity is 0.107. The fitted-plane phase-linearity median is 0.275, below the random-plane median of 0.442. Unit-state residuals give the same conclusion.

Updates show much stronger chronological organization. Raw updates have a median phase-linearity R² of 0.741, versus 0.398 after full time shuffling and 0.742 in random planes. Unit updates give 0.780 ordered, 0.406 shuffled, and 0.731 random-plane. Ordered phase linearity beats full shuffling in all 80 model × phase × representation comparisons, but the fitted plane beats the median random plane in only 42 of 80. The update sequence is smooth in high-dimensional space without selecting a special shared rotation plane.

The apparent update rotations are usually arcs or reversals. Raw updates span 0.548–4.429 total turns, 0.502–0.891 net turns, and monotonicity 0.219–0.946. Deep `combined_margin` raw updates accumulate 4.429 total turns but only 0.583 net turns, with monotonicity 0.219. Deep stable unit updates similarly have 3.614 total and 0.546 net turns, with monotonicity 0.189. Centering a smooth, roughly straight update sequence around its temporal mean naturally produces a half-turn-like angle sweep; the random-plane control shows that this effect is not specific to the PCA axes.

Given cells show similar update chronology. For example, `combined_margin` raw-update phase linearity during iterations 16–64 is 0.879 on blanks and 0.881 on givens. The temporal sweep is therefore not specific to cells being solved.

The representative paths agree with the metrics: the selected median cells form lines, U-shaped arcs, and deep zig-zag or backtracking paths rather than repeated coils. The plot selection is based on median phase-linearity score, so these are representative rather than hand-picked for visual appeal.

## Predictive rotation and spiral tests

Rotation beats both translation and the same-parameter cubic with both 95% intervals above zero in 11 of 80 ordered comparisons. These cases are scattered across models, phases, and representations. Flexible periodic coefficients can therefore capture some local held-out curvature, but there is no common frequency or phase regime.

The clearest local result is `late_state_ce` raw state at iterations 64–128, reported above. A second illustrative result is stable deep raw updates: a 0.25-cycle rotation beats translation by 0.437 [0.301, 0.567] but beats the cubic by only 0.023 [0.006, 0.040]. Its fitted-plane phase linearity is 0.746, essentially equal to 0.747 in random planes. Both results are better described as smooth local curvature than as a shared helix.

Spiral prediction is nearly uniformly negative against translation: 78 of 80 comparisons have intervals below zero, one is ambiguous, and one is positive. The single positive result is stable deep unit state, where a 0.25-cycle spiral beats translation by 0.623 [0.556, 0.688] and the quintic by 0.936 [0.930, 0.941]. The geometric path has radius CV 0.0041, total turns 0.0095, and net turns 0.00027. The sinusoidal basis extrapolates a gently curved path well, but the path does not rotate.

The cubic and quintic controls are poor suffix extrapolators in several settings. Periodic-versus-polynomial skill can approach one even after time shuffling, while periodic-versus-translation becomes strongly negative. The translation comparison and time-order controls are therefore more informative than the same-degree-of-freedom comparison by itself. Frequency labels also need caution: 67 of 80 spiral selections hit the maximum three-cycle grid value, with the fit objective still improving at the boundary. Those frequencies are not identified.

## Cyclic digit order

Digit centroids use unit-normalized token vectors and are computed separately on fit and held-out puzzles. Their representational-distance matrix is compared with every unoriented nine-digit cycle. At t=1024, the natural order remains nonspecific:

| model | natural cycle correlation / exact tail fraction | fit-selected cycle correlation / held-out rank | representative fit-selected cycle |
| --- | ---: | ---: | --- |
| `stable_plain` | -0.067 / 0.859 | 0.176 / 1 | `1-7-5-4-2-9-3-6-8` |
| `collapsed_plain` | 0.186 / 0.143 | 0.518 / 2 | `1-2-8-9-3-4-7-6-5` |
| `late_state_ce` | -0.171 / 0.912 | 0.529 / 1 | `1-2-6-3-8-4-9-7-5` |
| `combined_margin` | -0.003 / 0.452 | 0.162 / 12 | `1-4-3-5-9-8-2-7-6` |

Across five endpoints and four models, we tested centroids both before and after removing the output head's eight digit-contrast directions. That removal centers the nine output-weight rows and projects states perpendicular to their row space; it does not remove all digit information. Natural order has exact tail fraction below 0.05 in 0 of 40 tests. A fit-selected nonnumeric cycle has tail fraction below 0.05 in all 40 tests. The cycles are model- and endpoint-specific, and an unoriented cycle is equivalent under reversal and cyclic rotation. Removing those directions leaves the t=1024 results almost unchanged, so the model-specific organization is not confined to the output head's direct digit contrasts.

Planarity does not rescue the numeric interpretation. At t=1024, the train-fitted digit plane explains 0.401, 0.740, 0.622, and 0.755 of held-out centroid energy, yet the natural cycle remains nonspecific. The output-head rows themselves are nearly equidistant, with pairwise-distance CV 0.014–0.020, and have natural-order tail fractions 0.675–0.919. This is consistent with exchangeable class geometry rather than a numeric circle.

Actual prediction changes also reject the natural order: none of 11 endpoint/model rows has a natural adjacency tail fraction below 0.05. A train-selected order generalizes for only two rows, both early `late_state_ce` endpoints. Static digit geometry therefore should not be interpreted as digits winding around the trajectory in time.

## Correctness transitions, margin, and confidence

All four models reach 100% held-out puzzle accuracy at iteration 128. The stable and combined models have no later flips; `late_state_ce` has nine gains and no losses during 64–128, then none. The collapsed model resumes flipping: during 512–1024 it has 324 wrong→correct events across 21 puzzles and 647 correct→wrong events across 24 puzzles.

Transitions are output-boundary crossings accompanied by a local motion peak. Across models, the pooled wrong→correct median margin moves from -0.284 to -0.229 one iteration before the event to 0.308–0.431 at the new state; incoming update norm rises from 9.78–10.97 to 12.34–15.41. Correct→wrong margins move from 0.110–0.156 to -0.111–-0.221. These event-aligned q10/q90 bands pool correlated cell events and are descriptive, not puzzle-level confidence intervals.

Maximum-class confidence is not correctness confidence. During the collapsed model's deep phase, 90,016 wrong→wrong events have per-puzzle median target margin -1.994 while maximum-class confidence is 0.757. The overall margin median remains positive because 620,693 correct→correct events have median margin 138.7 and confidence 1.0. The negative p10 in the table exposes the failing tail that the median hides.

Correctness events do have an early association with update angle under the current cellwise circular-shift null: 18 of 23 exploratory phase × transition tests have nominal `p ≤ 0.05`. None survives a strict Bonferroni threshold of `0.05 / 23`, and the null does not match iteration, so this result cannot separate angle from ordinary temporal progress. The collapsed model's deep gains and losses are explicitly not aligned: resultant 0.141 versus null median 0.179, `p=0.690`, for gains; 0.152 versus 0.219, `p=0.907`, for losses.

A held-out logistic probe asks whether the previous update `u[t-1]` predicts correctness at `t+1` beyond current margin, margin velocity, entropy, log iteration, and rating bucket. Geometry changes gain AUROC by +0.00055 to +0.00953 across models. Loss AUROC changes by -0.00009 to +0.00202. The update signal is mostly redundant with output history, and these exploratory AUROC differences do not have puzzle-level uncertainty.

## Limits

This is one balanced 50-puzzle sample and one checkpoint for each model configuration, not a training-seed study. The 25-puzzle held-out bootstrap quantifies puzzle variation within this sample. It does not quantify checkpoint, dataset, or training-run variation.

The periodic models share a fit-selected frequency but allow every cell its own coefficients. A predictive win does not imply a common axis or phase. Random-plane comparisons use 12 planes and shuffle comparisons use 24 repetitions. The same-parameter polynomial baselines can extrapolate badly, and time reversal is not a falsification test for reversible curve families.

Event curves pool cells and iterations; event-phase tests use an iteration-unmatched null; exact cycle tail fractions measure specificity among 20,160 orders for one aggregate held-out RDM, not puzzle-sampling uncertainty. The many phase, representation, and endpoint comparisons are exploratory. High phase-step resultant is not used as evidence by itself because near-zero angle changes can also produce a resultant near one.

Within those limits, the consistent interpretation is smooth, compact, phase-dependent recurrent dynamics with strong translation and update-direction sweeps, plus stable model-specific digit organization. The evidence does not support a single shared natural-number helix.

## Artifacts and reproducibility

Numerical artifacts:

- [`metrics.json`](metrics.json): configuration, exact split, checkpoint paths, endpoint accuracy, fit-selected frequencies, and artifact manifest.
- [`basis_controls.csv`](basis_controls.csv): held-out explained variance, random subspaces, and state/update subspace overlap.
- [`phase_geometry.csv`](phase_geometry.csv): translation, turns, monotonicity, phase linearity, spiral scores, validity, givens, and time/random-plane controls.
- [`predictive_models.csv`](predictive_models.csv): held-out suffix scores for translation, rotation, spiral, cubic, quintic, and time controls.
- [`digit_cycles.csv`](digit_cycles.csv): exact natural and fit-selected cycle tests, nullspace results, output-head geometry, and prediction-change adjacency.
- [`correctness_transitions.csv`](correctness_transitions.csv), [`event_aligned_curves.csv`](event_aligned_curves.csv), [`event_phase_alignment.csv`](event_phase_alignment.csv), and [`transition_probes.csv`](transition_probes.csv): correctness, confidence, margin, event timing, angle alignment, and predictive probes.

Plots:

- [`phase_geometry.png`](phase_geometry.png): state translation and phase-linearity controls.
- [`predictive_helix_models.png`](predictive_helix_models.png) and [`predictive_spiral_models.png`](predictive_spiral_models.png): periodic suffix prediction on symlog axes.
- [`representative_cell_paths.png`](representative_cell_paths.png): representative held-out state and update paths.
- [`digit_cycle_controls.png`](digit_cycle_controls.png) and [`digit_centroid_projections.png`](digit_centroid_projections.png): exact cycle scores and held-out digit-centroid planes.
- [`correctness_event_dynamics.png`](correctness_event_dynamics.png) and [`correctness_transition_summary.png`](correctness_transition_summary.png): event-aligned trajectories, phase counts, and probe deltas.

Code:

- [`analyze_dynamics.py`](analyze_dynamics.py): complete analysis and rendering.
- [`modal_dynamics.py`](modal_dynamics.py): detached H200 runner with persistent output volume.
- [`test_dynamics.py`](test_dynamics.py): seven tests covering splits, cycle enumeration, phase metrics, state centering, predictive rotation, transition coding, and per-cell regression.
- [`progress.log`](progress.log): the completed four-checkpoint run, 166.8 seconds end to end.

Reproduce the tests and launch the detached analysis from the repository root:

```bash
source venv/bin/activate
python -B -m unittest looping.trajectory_viz.helix_tests.dynamics.test_dynamics -v
modal run --detach looping/trajectory_viz/helix_tests/dynamics/modal_dynamics.py --examples-per-bucket 10 --seed 20260807
```

All seven tests pass. I opened and inspected all eight generated figures at rendered resolution; labels and panels are intact, and the representative paths match the numerical conclusions. The numerical CSV/JSON artifacts come from the recorded Modal run. The current source only changes event-curve gathering to an equivalent vectorized implementation and adds presentation-only symlog/spiral rendering after that run.
