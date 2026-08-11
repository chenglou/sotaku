# Sotaku number-helix controls

## Conclusion

The strong number-helix hypothesis is not supported. Sotaku has genuine, highly transferable digit-category geometry, and iteration, confidence, margin, state norm, and cell position are all linearly readable from the recurrent state. However, the intrinsic digit geometry does not privilege the natural cyclic order `1→2→…→9→1` on held-out puzzles. A supervised two-dimensional readout can make the natural order look circular, but it makes an arbitrary fixed digit order look just as circular. PCA and fixed random bases do not reveal the same circle.

The result is evidence against a geometry whose natural digit cycle is intrinsic, stable across healthy checkpoints, and robust to puzzle, position, and state confounds. It does not rule out every weaker use of the word “helix.” In particular, these tests evaluate cyclic digit geometry and candidate progression variables separately; they do not exhaustively search for a single three-dimensional nonlinear manifold that jointly encodes digit and iteration.

## Primary held-out result

The primary population is blank cells. Each geometry entry averages five rating-stratified whole-puzzle folds, with 40 train puzzles and 10 held-out puzzles per fold; the final solved-count column evaluates the full 50-puzzle sample once. `Full residual` starts from unit-normalized states, subtracts a train-only mean for each of 81 cell positions at each sampled iteration, then removes state norm, confidence, the top-1–top-2 probability gap, and entropy with a train-only linear model. Digit centroids give every puzzle equal weight. Values after `±` are population standard deviations across the five folds, matching the plots.

| checkpoint | natural first-harmonic fraction | exact natural-order percentile | natural circle-distance correlation | train/test digit-distance correlation | sampled puzzles correct at iteration 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable | 0.240 ± 0.005 | 0.163 ± 0.039 | -0.068 ± 0.011 | 0.915 ± 0.040 | 50/50 |
| collapsed | 0.231 ± 0.005 | 0.293 ± 0.032 | -0.101 ± 0.008 | 0.904 ± 0.037 | 7/50 |
| late-state | 0.234 ± 0.004 | 0.220 ± 0.026 | -0.132 ± 0.012 | 0.868 ± 0.034 | 49/50 |
| combined | 0.244 ± 0.004 | 0.362 ± 0.032 | -0.032 ± 0.007 | 0.917 ± 0.034 | 50/50 |

The first-harmonic fraction is the held-out digit-centroid energy in the natural cosine/sine plane divided by total digit-centroid energy. A regular eight-dimensional nine-class simplex has fraction `2/8 = 0.25`; none of the four checkpoints exceeds that reference in the primary test. The exact percentile enumerates all 20,160 cyclic digit orders modulo rotation and reflection and evaluates train/test centroid agreement. A percentile above 0.5 would favor the natural order over a typical alternative. All four values are below 0.5, with fold ranges of 0.124–0.236, 0.235–0.329, 0.189–0.261, and 0.317–0.407 respectively. The negative circle-distance correlations are also inconsistent with natural neighbors being closer in representation space.

The high train/test digit-distance correlations show that the negative result is not caused by missing or noisy digit structure. The category geometry is real; the natural cycle is the part that fails.

## What the controls show

### Destroyed labels preserve trajectories but destroy transfer

The label controls independently rename digits per puzzle, shuffle digit labels among cell trajectories within each position, or assign class-count-preserving random labels fixed for an entire puzzle-cell trajectory. They preserve temporal strands and much of the sampling structure. Their train/test digit-distance correlations center near zero, while the real labels give 0.868–0.917. The real natural first-harmonic fractions, 0.231–0.244, are nevertheless ordinary within the destroyed-label distributions rather than unusually large. This separates a strong categorical signal from a natural-order signal.

### A fitted circle is not evidence for an intrinsic circle

Cross-validated phase cosine is nearly perfect for both a fitted natural order and a fitted arbitrary fixed order:

| checkpoint | fitted natural order | fitted arbitrary order | PCA basis | mean Haar-random basis |
| --- | ---: | ---: | ---: | ---: |
| stable | 0.998 | 0.997 | 0.092 | about 0.36 |
| collapsed | 0.984 | 0.960 | 0.196 | about 0.36 |
| late-state | 0.991 | 0.982 | 0.136 | about 0.36 |
| combined | 0.991 | 0.992 | 0.320 | about 0.36 |

This is the expected behavior of a supervised readout from linearly separable classes: the readout can assign a circle to any requested class order. The output-head harmonic basis also has phase cosine around 0.996–0.997 because it directly maps states to digit logits. The projection comparison makes the control concrete: the supervised natural and arbitrary projections both draw clean polygons, whereas PCA and representative fixed random projections do not.

### Digit identity survives position, time, confidence, and decoder controls

On unit states, held-out balanced accuracy is 0.825–0.874 for true digit and 0.936–0.970 for predicted digit. Log iteration has held-out linear `R²` 0.884–0.952; confidence, the top-1–top-2 probability gap, and state norm have `R²` around 0.88–0.94. Signed true-logit margin has `R²` 0.50–0.56. Cell identity is also readable at 0.076–0.132 accuracy against `1/81 ≈ 0.012` chance. These variables therefore can produce visually organized trajectory plots.

After the exact cell-position-by-iteration mean is removed, position and iteration probes fall to chance or approximately zero `R²`, but true-digit accuracy remains 0.824–0.873. Removing confidence, the probability gap, entropy, and norm leaves true-digit accuracy at 0.823–0.873. Signed true-logit margin is label-dependent and therefore excluded from the primary residual; an explicit sensitivity residual that also removes it gives exact natural-order percentiles 0.173, 0.143, 0.244, and 0.424 for stable, collapsed, late-state, and combined, all below 0.5. Projecting out the centered output-head row span also leaves the true-digit accuracy in the same range. Sotaku therefore carries redundant categorical information beyond a direct decoder readout, but the residual category geometry still does not prefer natural cyclic order.

A position-and-iteration-only digit baseline is around 0.095 balanced accuracy, below `1/9 ≈ 0.111` chance. In the stricter position-transfer test, each fold holds out both puzzles and one of three interleaved cell-position groups. Pooling all 15 fold-by-position-group results per checkpoint, mean exact natural-order percentiles remain 0.178, 0.420, 0.182, and 0.296 for stable, collapsed, late-state, and combined checkpoints. Position structure does not explain the main result.

### Predicted labels, true labels, phases, and updates agree on the broad conclusion

Using predicted rather than true digits on all-iteration unit states gives exact percentiles 0.162, 0.244, 0.176, and 0.311. Restricting unit states to incorrect early blank cells also fails to produce a natural cycle: predicted-label percentiles are 0.119, 0.126, 0.248, and 0.263, and true-label percentiles are 0.171, 0.085, 0.145, and 0.198.

Across early, middle, late, and all-iteration phases, the healthy stable and late-state checkpoints remain below the natural-order null. Unit-update exact percentiles are 0.144, 0.651, 0.187, and 0.603 for stable, collapsed, late-state, and combined, with respective natural first-harmonic fractions 0.233, 0.260, 0.201, and 0.262. The larger collapsed and combined update ranks do not correspond to dominant two-dimensional components, and they do not recur in stable/late-state updates or the checkpoints’ state geometry.

### The collapsed checkpoint contains localized natural-order effects

The failing checkpoint is the main exception, and the exception is informative. Its raw-state primary exact percentile is 0.888 with natural first-harmonic fraction 0.379. Unit normalization reduces those values to 0.410 and 0.238; the full residual gives 0.293 and 0.231. Thus most of the all-iteration raw effect is tied to state magnitude and collapse dynamics.

Within the collapsed checkpoint’s late phase, unit states reach exact percentile 0.713 with fraction 0.279. Its lowest two confidence quartiles have true-label percentiles 0.862 and 0.758; the lowest two probability-gap quartiles have 0.815 and 0.810. Predicted-label strata do not show the same effect. This is evidence for a localized natural-order component in a failing, low-confidence regime. It is evidence against the stronger claim that a stable natural number helix is a shared mechanism of successful recurrent reasoning.

Given cells are another leakage-sensitive exception. Their all-iteration unit-state percentiles range from 0.475 to 0.792, compared with 0.165 to 0.410 for blank cells, even though their first-harmonic fractions stay near 0.25. Directly supplying a digit can preserve small order biases from the input interface, so blank cells are the appropriate primary population.

### Parameter matrices are close to an eight-dimensional class simplex

The nine digit rows of the initial encoder, recurrent feedback projection, and output head have effective ranks of about 7.77–7.99. Their leading two digit PCs explain only 0.26–0.31 of between-digit energy. Within each checkpoint, pairwise centered-kernel alignment among the three matrices is 0.983–0.997. This is predominantly high-dimensional category geometry, not a two-dimensional ring.

There is a small parameter-level natural-order anisotropy that should not be hidden: stable initial/feedback weights have exact order percentiles 0.928/0.985, and late-state feedback has 0.984. Their absolute natural first-harmonic fractions are only about 0.26, however, and the effect is absent from the corresponding output heads, the combined checkpoint’s interface, and the primary hidden-state tests. The weights can contain a weak natural-order bias without making natural order the dominant recurrent geometry.

## Checkpoints and sampled behavior

The run used the canonical checkpoints already referenced by the repository diagnostics:

- stable: `/outputs/model_baseline_lr2e3.pt`
- collapsed: `/outputs/model_baseline_lr2e3_clean_a.pt`
- late-state: `/outputs/looping/model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt`
- combined: `/outputs/looping/model_loop_stay_late_switch_margin_floor5_from39k.pt`

On the 50 held-out-control puzzles, stable solves 50/50 at iterations 256, 512, and 1024. Collapsed goes from 44/50 at 128 to 7/50 at 1024. Late-state solves 50/50 at 128, 256, and 512, then 49/50 at 1024. Combined solves 50/50 from 128 onward. These sampled outcomes reproduce the intended healthy/collapsed checkpoint distinction.

## Method

- Fifty test puzzles were balanced across five rating buckets and divided into five identical whole-puzzle folds for every checkpoint. This avoids the rating-ordered split confound in an earlier trajectory diagnostic.
- States were sampled at iterations 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, and 1024. Phase summaries use early 1–16, middle 32–128, and late 256–1024 snapshots.
- All nuisance fits, readout bases, centering terms, quartile thresholds, and probes were fit only on training puzzles and evaluated on held-out puzzles.
- The suite compares raw states, unit states, unit updates, cell-position/time residuals, full confound residuals, and output-head-span-removed states; true and predicted digit labels; blank, given, and incorrect-early cells; phase and confidence/margin strata; and jointly held-out cell-position groups.
- Intrinsic order tests enumerate every cyclic digit order. Projection controls include PCA, an output-head basis, a fitted natural-order basis, a fitted arbitrary-order basis, and 128 fixed Haar-random two-dimensional bases per fold. Label controls use 20 repeats per null type and fold.
- Fold error bars are descriptive population standard deviations, not confidence intervals. The puzzle is the held-out unit, even though each puzzle contributes many cell trajectories.

## Limits

This run uses one checkpoint per named regime and 50 held-out puzzles, not multiple independent training seeds. It samples through iteration 1024, so it does not test later 2048/4096 behavior. The exact-order, label-null, position-transfer, and phase analyses are stronger than a visual projection, but they do not constitute a formal test of every nonlinear or locally varying three-dimensional manifold. The high linear readability of iteration and confidence shows that candidate progression axes exist; the failed necessary ingredient is a robust intrinsic natural digit cycle in the same states.

## Artifacts and inspection

The final artifacts are `metrics.json`, `run.log`, `intrinsic_metrics.csv`, `probe_metrics.csv`, `basis_controls.csv`, `label_null_controls.csv`, `position_transfer.csv`, `confidence_margin_strata.csv`, `weight_geometry.csv`, and `sample_accuracy.csv`. The implementation is in `analyze_controls.py`, the detached runner is `modal_controls.py`, and `test_analyze_controls.py` contains nine passing unit tests. Review also exercised deterministic toy-state collection, a synthetic full `analyze_model` call, and all plotting functions separately.

I opened and inspected every final PNG at rendered resolution. The checkpoint panels, reference lines, legends, labels, and plotted exceptions are visible without apparent clipping or invalid scales:

- [`natural_order_checkpoints.png`](natural_order_checkpoints.png): phase-by-checkpoint intrinsic order results, including the collapsed late-phase spike.
- [`confounds_and_decoder_span.png`](confounds_and_decoder_span.png): unit-state, position/time-residual, full-residual, and decoder-span comparisons for natural-order metrics and digit/position probes.
- [`axis_probe_r2.png`](axis_probe_r2.png): held-out scalar readability of log iteration, confidence, probability margin, true-logit margin, and log state norm from unit states.
- [`heldout_cell_positions.png`](heldout_cell_positions.png): simultaneous puzzle and position-group transfer.
- [`label_and_basis_nulls.png`](label_and_basis_nulls.png): destroyed-label and random-basis controls.
- [`projection_comparison.png`](projection_comparison.png): supervised natural and arbitrary polygons beside PCA and fixed-random projections.
- [`confidence_margin_strata.png`](confidence_margin_strata.png): within-iteration confidence and margin quartiles, including the collapsed low-confidence exception.
- [`weight_geometry.png`](weight_geometry.png): interface effective rank, leading-PC energy, and small natural-order anisotropies.

The suite also vendors the prior `existing_trajectory_geometry_v2.json` result for comparison. That diagnostic found compact and smooth token updates but poor whole-board transfer. The new controls use truly held-out, rating-stratified puzzles and distinguish stable categorical geometry from natural cyclic order; their conclusion is compatible with the earlier global-PCA and trajectory-control plots, which look like rays or fans rather than a shared helix.
