# Held-out linear and cyclic tests of Sotaku's digit geometry

## Conclusion

Sotaku has strong, stable digit-category geometry, but these tests do not support a shared natural-number helix. Linear, one-turn cyclic, and combined three-degree-of-freedom digit codes all predict held-out hidden-state directions. An unrestricted eight-degree-of-freedom categorical code predicts much more, however, and the combined code usually captures about the `3/8` fraction expected from its rank alone. More importantly, the natural mapping `1,2,…,9` is not consistently exceptional when the nine labels are globally permuted. All 80 tests in the analysis-defined hidden-state natural-order family have Holm-adjusted `p = 1.00`.

The collapsed checkpoint is the only nominal exception: its raw combined-code permutation values are `p = .0303–.0481`, depending on label and output-head projection. Those values do not survive family-wise correction, the natural linear component is ordinary after conditioning on the cyclic component (`p = .252–.320`), and the pattern is absent from the healthy checkpoints. This agrees with the earlier finding of a localized natural-order deformation during collapse, not a geometry shared by successful recurrent solving.

The strongest interpretation is therefore categorical rather than numerical. A low-dimensional probe can compress some of the nine digit regions into a chosen ramp and circle, but natural numerical order is not reliably distinguished from alternative label orders across checkpoints and does not survive family-wise correction.

## Operational definition

The arithmetic “number helix” result motivates a representation with a linear number coordinate plus periodic Fourier coordinates. For Sudoku's nine symbols, the fixed one-turn analogue used here is

`[(d - 5) / sqrt(20/3), cos(2π(d - 1)/9), sin(2π(d - 1)/9)]`, for `d ∈ {1,…,9}`.

The analysis fits four multivariate models to 128-dimensional cell responses:

- `linear`: the centered digit ramp, one degree of freedom;
- `cyclic`: one fixed turn across digits `1→…→9→1`, two degrees of freedom;
- `helix`: the linear and cyclic columns together, three degrees of freedom;
- `categorical`: unrestricted digit identity, eight degrees of freedom.

The linear ramp is correlated with the fixed Fourier columns over only nine samples: its correlations with cosine and sine are `-.274` and `-.752`. Raw linear and cyclic scores therefore cannot establish two distinct components. The permutation family also tests cyclic gain after the linear code and linear gain after the cyclic code. A convincing helix should make both additions exceptional under the same natural label order.

This is a nine-class linear test, not a claim that the states literally follow the multi-period arithmetic helix described in [*Language Models Use Trigonometry to Do Addition*](https://arxiv.org/abs/2502.00873). The temporal trajectory question is handled separately in the [dynamics analysis](../dynamics/REPORT.md).

## Held-out design

- The sample contains 100 canonical test puzzles: 20 from each rating bucket `0`, `1–2`, `3–10`, `11–50`, and `51+`.
- Five folds hold out complete puzzles, with exactly four puzzles from every rating bucket in each fold. The same folds are used for every checkpoint, representation, label, and permutation.
- The response uses originally blank cells at iterations `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024`. Iteration zero is the encoder state.
- At each puzzle and iteration, the mean over all 81 cells is removed before each cell vector is normalized. The primary analysis therefore tests direction rather than board-wide translation or state magnitude.
- A second representation also removes the output head's eight digit-contrast directions before normalization: center the nine output-weight rows and project the cell state perpendicular to their row space. Historical artifacts call this `output_null`. It removes linear directions that distinguish output digits, not the entire output head or every possible representation of digit identity.
- The primary nuisance model contains rating-bucket indicators, blank fraction, exact iteration indicators, and exact cell-position indicators. These variables are label-free. Each puzzle has equal total weight regardless of its number of blanks.
- Optional certainty sensitivities use thresholds fixed in code before the final rerun to bin maximum confidence and top-one/top-two logit margin. A target-informed sensitivity additionally bins true-class probability and true-vs-best-wrong margin. Binning prevents nearly constant late-iteration confidence from generating unstable held-out slopes.
- Held-out partial `R²` is `1 - SSE_full / SSE_nuisance` and is never clipped at zero.
- Weighted multivariate OLS uses an SVD relative singular-value cutoff of `1e-8`. Unit tests compare the sufficient-statistic permutation implementation with direct cross-validation, including a nearly redundant nuisance design.
- The plotted 95% intervals use 2,000 complete-puzzle residual resamples within rating buckets. Probe fits and folds remain fixed, so these are conditional descriptive intervals rather than population confidence intervals.
- Natural-order tests use 9,999 global label permutations, with one mapping shared across all cells, puzzles, iterations, and folds. The one-sided alternative is that the observed natural-order fit is larger. Plus-one Monte Carlo `p` values are corrected by Holm's method across 80 tests of the primary states and states with digit-contrast directions removed. Learned-parameter tests form a separate 36-test family.

An earlier preliminary trajectory fit used a contiguous split even though the balanced loader appends rating buckets in blocks. This analysis replaces that difficulty-confounded split with the within-bucket whole-puzzle folds above.

## Primary held-out fits

The table reports pooled partial `R²` after the primary nuisance controls. “Fraction” is helix gain divided by categorical gain. The last column is the unadjusted global label-permutation value for the combined helix; every corresponding Holm-adjusted value is `1.00`.

| checkpoint | label | linear | cyclic | helix | categorical | fraction | natural helix `p` |
|---|---|---:|---:|---:|---:|---:|---:|
| stable plain | true | .061 | .113 | .177 | .480 | .368 | .6494 |
| stable plain | predicted | .069 | .128 | .201 | .544 | .369 | .6291 |
| collapsed plain | true | .057 | .138 | .195 | .418 | .467 | .0376 |
| collapsed plain | predicted | .064 | .149 | .214 | .460 | .464 | .0303 |
| later-iteration training | true | .043 | .086 | .125 | .361 | .346 | .9224 |
| later-iteration training | predicted | .051 | .101 | .146 | .419 | .348 | .9148 |
| combined margin | true | .035 | .066 | .102 | .273 | .375 | .5027 |
| combined margin | predicted | .041 | .079 | .121 | .323 | .375 | .4941 |

The conditional 95% intervals for true-digit helix `R²` are `[.166, .186]`, `[.186, .203]`, `[.117, .131]`, and `[.0958, .108]` for stable, collapsed, late-state, and combined checkpoints. The fixed-sample held-out gains and conditional intervals are positive. What fails is the shared natural-order interpretation: a three-dimensional code will generally recover some structure from an eight-dimensional nine-class arrangement, and natural order is not reliably exceptional across checkpoints or after family-wise correction. [The fit summary](heldout_fit_summary.png), [permutation nulls](natural_order_permutations.png), and [geometric adequacy plot](geometric_adequacy.png) show the distinction.

Across the 80 natural-order tests, the smallest raw value is the collapsed predicted-digit helix result above, `p = .0303`; all adjusted values are `1.00`. For that same row, cyclic gain after the linear code has raw `p = .0474`, while linear gain after the cyclic code has `p = .2518`. The collapsed result therefore does not supply both components required by the helix interpretation even before correction. The other three checkpoints have raw combined-code values of `.4505–.9224` across both representations. [The complete adjusted test matrix](natural_order_test_matrix.png) makes the family-wide result explicit.

## Iteration, position, confidence, and margin

Iteration is important but does not turn the categorical geometry into a natural helix. Before digit labels are added, exact iteration indicators improve held-out prediction of primary state directions by partial `R² = .146–.220`; exact cell position adds another `.0255–.0490`. Both are included in every pooled digit result above.

True-digit fits change substantially over recurrence:

| checkpoint | peak helix iteration / `R²` | peak categorical iteration / `R²` | iteration 1024 helix / categorical |
|---|---:|---:|---:|
| stable plain | 1024 / .330 | 1024 / .942 | .330 / .942 |
| collapsed plain | 512 / .476 | 256 / .865 | .400 / .716 |
| later-iteration training | 32 / .250 | 128 / .737 | .148 / .527 |
| combined margin | 64 / .257 | 32 / .686 | .0527 / .163 |

The stable checkpoint accumulates increasingly separable digit regions. The other checkpoints peak earlier, and the combined-margin direction geometry becomes much less digit-specific even while its predictions remain correct. The categorical curve remains far above the three-degree-of-freedom curve throughout the informative part of each trajectory. See [per-iteration fits](iteration_profiles.png).

Prediction accuracy, confidence, and margin confirm that these regimes differ. At iteration 1024 the stable, late-state, and combined checkpoints are correct on all sampled blank cells; collapsed plain has fallen to `.839` accuracy despite `.925` mean maximum confidence. The corresponding confidence and weakest-tail target/decision margins are shown in [confidence and margin trajectories](confidence_margin_trajectories.png).

The corrected fixed-bin sensitivities are numerically stable and leave the interpretation unchanged:

| checkpoint | true-digit helix `R²`: position → decision bins → target bins | predicted-digit helix `R²`: position → decision bins → target bins |
|---|---:|---:|
| stable plain | .177 → .184 → .185 | .201 → .209 → .209 |
| collapsed plain | .195 → .199 → .199 | .214 → .220 → .221 |
| later-iteration training | .125 → .128 → .128 | .146 → .150 → .150 |
| combined margin | .102 → .106 → .106 | .121 → .125 → .125 |

The label-free confidence and decision-margin bins reduce held-out state-direction SSE by `.0271–.0436` beyond iteration and position, while the target-informed bins change it by only `-.00026–.00124`. The target-adjusted helix score differs from the decision-adjusted score by at most `.00131`. Every decision-adjusted training fold has rank 106 of 108 columns; target-adjusted ranks are 117–119 of 123, with exact one-hot/intercept redundancies removed by the SVD solve. No fold has a non-finite or extreme held-out loss. Because changing the nuisance model can also change the partial-`R²` denominator, the small upward shifts are not evidence that certainty creates a helix. They show that the negative natural-order conclusion is insensitive to this coarse adjustment. See [sequential nuisance contributions](nuisance_controls.png) and [the helix-fit sensitivity](certainty_sensitivity.png).

## True digit versus predicted digit

Predicted-digit fits are consistently larger than true-digit fits in the pooled sample, partly because predictions and solutions agree on most late cells. Conditioning on incorrect blank-cell snapshots separates the labels:

| checkpoint | incorrect cell-snapshots | state helix `R²`, true / predicted | helix `R²` after removing digit contrasts, true / predicted |
|---|---:|---:|---:|
| stable plain | 13,885 | .0146 / .0590 | .0029 / .0305 |
| collapsed plain | 13,967 | .0149 / .0558 | .0039 / .0309 |
| later-iteration training | 13,388 | .0153 / .0715 | .0051 / .0491 |
| combined margin | 13,686 | .0167 / .0720 | .0062 / .0493 |

On errors, the representation follows the model's current predicted category much more strongly than the puzzle's true digit. This supports a recurrent hypothesis/category representation rather than a fixed numerical coordinate for the solution. The wrong-only analysis is conditional on making an error and is descriptive; it is not a randomized comparison.

## Output-head and update controls

Removing the eight digit-contrast directions defined above does not make natural order exceptional:

| checkpoint | true-digit helix `R²` / raw `p` | predicted-digit helix `R²` / raw `p` |
|---|---:|---:|
| stable plain | .163 / .7111 | .185 / .6832 |
| collapsed plain | .188 / .0481 | .204 / .0398 |
| later-iteration training | .120 / .9131 | .141 / .9027 |
| combined margin | .0959 / .4647 | .114 / .4505 |

All Holm-adjusted values after that removal are again `1.00`. The projection retains `88.5%`, `96.5%`, `98.8%`, and `98.2%` of cell-centered state energy at iteration 1024 for stable, collapsed, late-state, and combined checkpoints. Sotaku therefore carries redundant digit-category information outside the direct readout contrast span, but that information still does not privilege natural numerical order. See [the permutation controls after removing digit contrasts](output_null_order_permutations.png).

One-step update directions also contain digit information. Their pooled true-digit helix/categorical partial `R²` values are `.155/.428`, `.153/.310`, `.041/.133`, and `.0318/.0848` for stable, collapsed, late-state, and combined checkpoints. These update fits are descriptive in this run; no additional update permutation family was added. The cell-centered update RMS norms at iteration 1024 remain `2.93–6.62`, no sampled update at that iteration is below `1e-8`, and the update/state RMS ratio has fallen to `.00094–.00142`. Direction-normalized late updates are therefore defined, but their small relative scale requires caution. See [response scale checks](response_scales.png) and the independent [temporal geometry report](../dynamics/REPORT.md) for exact update-cycle tests.

## Learned parameters and fitted-axis shape

The input digit columns, prediction-feedback columns, and output-head rows behave like high-dimensional class interfaces rather than a natural-order helix. Across all four checkpoints, raw parameter-fit `R²` lies near the rank-only references: `.123–.136` for a one-dimensional linear code, `.242–.264` for a two-dimensional cyclic code, and `.357–.390` for the three-dimensional combination. The corresponding exchangeable eight-dimensional references are `1/8`, `2/8`, and `3/8`.

Among 36 parameter order tests, the smallest unadjusted value is `p = .0150` for the stable checkpoint's prediction-feedback cyclic fit; its Holm-adjusted value is `.540`. No learned-parameter test is significant after family-wise correction. The [parameter plot](parameter_geometry.png) shows that the observed raw fits sit close to their permutation means.

The fitted coefficient blocks are stable across folds, but the fold training sets overlap substantially, so that stability is descriptive rather than an independent replication. Their spatial axes are not a clean helix. The mean cyclic minor/major singular-value ratio is about `.55–.58` for stable, late-state, and combined states, and the linear axis has `.61–.66` of its energy inside the fitted cyclic span. Collapsed states have a more nearly circular cyclic block, ratio about `.90`, but retain the same large linear-axis overlap and are the failing regime. Stable recovery of an arbitrary three-dimensional regression subspace is not evidence that the natural numeric parameterization generated the original class geometry.

## Relation to the existing diagnostics

These results reinforce the preceding analyses from a different statistical direction:

- [Number-helix controls](../controls/REPORT.md) find transferable categorical geometry but ordinary natural first-harmonic rank under all exact cyclic orders. Supervised natural and arbitrary orders both draw clean circles.
- [Per-cell temporal geometry](../dynamics/REPORT.md) finds real smooth, low-dimensional dynamics and model-specific nonnumeric digit cycles, but no shared natural digit cycle in states before or after removing digit contrasts, output rows, or actual prediction changes.
- [Visual geometry tests](../visual/REPORT.md) find high-rank digit centroids and no pooled natural cycle, with a localized collapse-specific update exception.
- [Whole-board projection controls](../../controls/REPORT.md) find smooth rays and arcs whose visible shape changes with the projection, rather than a repeated helix.

The present analysis adds difficulty-balanced whole-puzzle cross-validation, simultaneous linear and cyclic regressors, nested component tests, global label-permutation inference, confidence/margin sensitivities, exact position controls, and learned-parameter nulls. Every route leads to the same distinction: digit identity is real; natural digit order is not the organizing principle.

## Limits

- There is one checkpoint per named training regime, not independent training seeds. The checkpoints and test dataset were already examined in related work, so this is not a fresh preregistered confirmation.
- The fixed one-turn period-nine code is a precise linear analogue for nine symbols, not an exhaustive search over nonlinear, locally varying, or multiple-period manifolds.
- Board-wise centering and per-cell unit normalization deliberately remove global translation and magnitude. The separate scale summaries retain those quantities, but the primary geometric claim is about normalized directions.
- Removing the output head's digit-contrast directions does not remove every representation of digit identity. Redundant linear information or a nonlinear readout could still recover structure.
- Update fits in this run do not receive a separate permutation family. Existing exact update-cycle tests provide the order control.
- Confidence and margin bins are coarse additive sensitivities with a common bin effect across iterations; they do not remove within-bin variation or confidence-by-iteration interactions. The variables are downstream model outputs, and the target-informed version also conditions on the answer, so neither design is causal or confirmatory. No bootstrap or permutation inference is attached to these sensitivities.
- Conditional bootstrap intervals do not refit probes, and cross-validation training sets overlap. Natural-order permutation values, not the interval widths, carry the inferential claim.
- Wrong-only fits condition on model errors. They identify which label better describes those states but do not estimate a causal effect.
- These are observational geometry tests. No activation patching or other causal intervention was performed.
- Fitted-axis ratios and overlaps depend on the chosen response metric and normalization; they are descriptive shape checks rather than invariant topological quantities.
- Sudoku is invariant to a global permutation of digit symbols. The task does not require a natural numeric order, so the negative result is compatible with successful solving.

## Artifacts and reproduction

- Complete configuration, folds, puzzle hashes, checkpoints, fold losses, conditional intervals, geometry descriptors, and statistics: [`results.json`](results.json)
- Saved permutation distributions: [`permutation_nulls.npz`](permutation_nulls.npz)
- Compact tables: [`pooled_fits.csv`](pooled_fits.csv), [`iteration_fits.csv`](iteration_fits.csv), [`nuisance_controls.csv`](nuisance_controls.csv), [`certainty_sensitivity.csv`](certainty_sensitivity.csv), [`wrong_only_fits.csv`](wrong_only_fits.csv), and [`parameter_geometry.csv`](parameter_geometry.csv)
- Analysis: [`analyze_number_helix.py`](analyze_number_helix.py), [`statistics_core.py`](statistics_core.py), [`trajectory_data.py`](trajectory_data.py), and [`parameter_geometry.py`](parameter_geometry.py)
- Plotting: [`plot_statistics.py`](plot_statistics.py)
- Modal entrypoint and log: [`modal_statistics.py`](modal_statistics.py), [`run.log`](run.log)
- Tests: [`test_analysis_design.py`](test_analysis_design.py), [`test_statistics_core.py`](test_statistics_core.py), [`test_trajectory_data.py`](test_trajectory_data.py), and [`test_parameter_geometry.py`](test_parameter_geometry.py)

The final plots were opened at full resolution and inspected for clipping, scale, labels, and misleading connections. The iteration figure was rerendered with row-specific x labels removed so panel titles and labels do not overlap. The corrected certainty figures use the fixed-bin results, and their legends do not cover the data or zero reference.

```bash
source venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s looping/trajectory_viz/helix_tests/statistics -p 'test_*.py'
modal run --detach looping/trajectory_viz/helix_tests/statistics/modal_statistics.py --examples-per-bucket 20 --permutations 9999 --seed 20260807
```
