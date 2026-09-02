# Number-helix geometry tests

## Conclusion

Sotaku has a stable, linearly readable categorical digit representation, but these tests do not find a shared intrinsic 1→…→9 number helix. A supervised projection can put held-out states and updates on a clean circle or helix, with pooled test sector accuracy of 67.3–72.4% for unit-normalized states and 38.0–57.0% for unit-normalized updates. That result establishes digit readability: the circle and helix are chosen targets of the readout, not geometry discovered in the hidden space.

The label-free and original-space checks point the other way. Across four checkpoints, the first three train-PCA axes capture only 15.5–37.8% of held-out unit-state or unit-update energy. The nine held-out digit centroids usually have effective rank 6–8, their pairwise distances correlate weakly with the distances of an ideal natural-order circle or helix, and the natural digit cycle is not unusually short among all 20,160 possible cycles. The shortest cycle selected on training puzzles transfers extremely well to test puzzles, but its order changes across checkpoints and between states and updates. Sotaku therefore has reproducible digit-specific geometry without a reproducible numeric ordering.

There is one localized exception worth retaining: collapsed-plain updates at iteration 256 become substantially lower-dimensional and put the natural cycle in the shortest 1.51% of cycles. Related late collapsed-state slices also give low natural-cycle percentiles. The ideal circle and helix correlations remain only about 0.25, the train-selected shortest order is not numeric order, the pooled collapsed-update percentile is 9.95%, and the pattern does not occur in the healthy checkpoints. This looks more like a collapse-specific deformation than a shared number helix.

## Relation to the existing diagnostics

The existing whole-board controls found a large state-norm direction and smooth but projection-dependent trajectories, with no shared helix. The token-feature diagnostics also showed that updates are much more compressible per cell than after flattening an entire board. This experiment therefore works at the per-cell level, analyzes originally blank cells only, and reports unit-normalized states and updates as its primary representations. Raw variants remain in the structured results, but they are not used for the headline geometric claim.

## Data and validation

- Checkpoints: stable plain, collapsed plain, later-iteration training, and combined margin.
- Data: 100 held-out Sudoku puzzles, balanced as 20 puzzles from each of the five rating buckets used by the trajectory diagnostics.
- Split: 50 train, 25 validation, and 25 test puzzles, stratified within rating bucket. All cells from a puzzle stay in one split.
- Population: cells that were blank in the input puzzle. Given cells are excluded because their input embeddings expose digit identity directly.
- Iterations: `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024`. A state is `h_t`; its update is `h_(t+1) - h_t`, including one extra recurrent step to form the update at 1024.
- Inference: the repository's canonical CUDA bf16-autocast path, including the initial encoder pass inside autocast.
- Display attributes: every 2D atlas reuses identical held-out points and colors them by true digit, predicted digit, prediction confidence, true-digit margin, recurrent iteration, and cell row/column. Digit panels use the same cyclic 1→…→9 hue order.
- Uncertainty: headline readout intervals use a stratified puzzle bootstrap, resampling puzzles within rating bucket rather than treating cells as independent observations.

PCA is fitted on training-puzzle vectors only. “Test PCA top 3” below is the fraction of held-out squared energy about the training mean captured by the first three training axes; it is not a refitted test-set variance fraction.

The supervised periodic readout first fits nine one-hot digit scores with ridge regression. Ridge strength is chosen on validation-puzzle one-hot MSE, which is invariant to a permutation of digit labels. Only after fitting are the nine scores composed with a fixed natural-order circle or helix code. The final metrics use the untouched test puzzles. This construction prevents validation from favoring the requested natural order, but it also means that a circular supervised plot cannot be used as evidence that a circle was already present.

## Intrinsic digit geometry

For each representation, the full 128-dimensional train and test digit centroids are analyzed directly. The natural-cycle percentile is the fraction of all 20,160 digit cycles whose test length is no greater than the natural 1→…→9 cycle length; lower values are more compatible with natural cyclic adjacency.

| Checkpoint | Representation | Test PCA top 3 | Centroid top 3 | Effective rank | Train/test distance r | Circle distance r | Helix distance r | Natural-cycle percentile |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stable plain | unit state | 34.4% | 45.4% | 7.66 | .977 | -.064 | .282 | 97.47% |
| stable plain | unit update | 24.5% | 46.7% | 7.50 | .990 | -.087 | .254 | 98.56% |
| collapsed plain | unit state | 37.8% | 49.4% | 7.12 | .962 | .031 | .045 | 16.62% |
| collapsed plain | unit update | 26.6% | 52.0% | 7.03 | .975 | .057 | .223 | 9.95% |
| later-iteration training | unit state | 32.6% | 45.2% | 7.64 | .899 | -.122 | -.046 | 40.63% |
| later-iteration training | unit update | 17.3% | 56.2% | 6.72 | .983 | -.076 | -.186 | 46.52% |
| combined margin | unit state | 34.2% | 47.8% | 7.46 | .976 | -.030 | .267 | 54.04% |
| combined margin | unit update | 15.5% | 57.7% | 5.95 | .966 | -.044 | .201 | 36.53% |

The high train/test distance correlations show that the negative natural-order result is not caused by noisy centroid estimates. A low-dimensional natural helix would instead predict a concentrated centroid spectrum, positive agreement with ideal natural-order distances, and an unusually short natural cycle. No pooled representation has all three properties; no pooled natural-cycle percentile is below 5%.

The train-selected shortest cycles make the distinction especially clear. Their orders include `1-3-5-7-8-4-2-9-6` for stable states, `1-6-9-2-4-5-7-3-8` for stable updates, and different orders again for every other checkpoint/representation. These cycles transfer into the shortest 0.005–0.055% of test cycles. There is stable categorical structure, but its adjacency is arbitrary with respect to numeric order. This is also consistent with Sudoku's digit-permutation symmetry: the task supplies category identity but no ordinal relation between, for example, 3 and 4.

## Supervised periodic readability

The table reports one readout fitted across all recorded iterations. Sector accuracy assigns each predicted 2D coordinate to its nearest natural-order digit sector; chance is 11.1%. The last two columns remove the centered eight-dimensional output-head row span before normalization and fitting.

| Checkpoint | Unit-state sector accuracy | Unit-update sector accuracy | State without output span | Update without output span |
|---|---:|---:|---:|---:|
| stable plain | 71.9% [67.6, 75.3] | 57.0% [52.2, 60.5] | 72.8% | 57.0% |
| collapsed plain | 67.3% [62.5, 71.1] | 50.9% [45.6, 55.4] | 68.6% | 51.1% |
| later-iteration training | 72.4% [68.1, 76.8] | 42.7% [39.9, 45.4] | 72.7% | 42.7% |
| combined margin | 72.0% [68.9, 75.3] | 38.0% [35.0, 40.5] | 72.1% | 38.8% |

Removing the output-head span changes sector accuracy by at most 1.25 percentage points. Digit identity is therefore distributed outside the exact linear decoder directions; the periodic plots are not merely a visualization of the output weights.

Readability also evolves differently for states and updates. State identity becomes very clear by iteration 128 and remains clear in the healthy late-state checkpoints, while a single pooled linear readout becomes progressively less able to decode late updates. This decline can reflect shrinking updates and changing update directions; it is not evidence that late updates lack all digit information.

| Checkpoint | State t16 / t128 / t512 / t1024 | Update t16 / t128 / t512 / t1024 |
|---|---:|---:|
| stable plain | 84.1 / 96.9 / 100.0 / 100.0% | 82.5 / 80.4 / 31.9 / 14.6% |
| collapsed plain | 84.4 / 96.4 / 91.5 / 64.7% | 78.7 / 76.3 / 20.3 / 12.4% |
| later-iteration training | 82.6 / 94.2 / 99.2 / 97.5% | 74.7 / 41.8 / 20.7 / 17.1% |
| combined margin | 81.4 / 100.0 / 100.0 / 98.4% | 75.5 / 28.5 / 20.6 / 15.1% |

The supervised circle coordinate has pooled test circular R² of .660–.706 for states and .200–.490 for updates. The natural helix target has pooled overall R² of .667–.708 for states and .225–.507 for updates. These are useful measures of linear digit readability under the chosen codes, not intrinsic helix scores. An exact cycle-order null applied to the readout centroids likewise finds no compelling natural-order-specific advantage over other target permutations.

## Localized late-collapse pattern

The strongest horizon-specific result is the collapsed-plain unit update at iteration 256:

- centroid top-three fraction: 82.39%
- centroid effective rank: 3.27
- natural-cycle percentile: 1.51%
- natural-circle distance correlation: .254
- natural-helix distance correlation: .261
- train-selected shortest order: `1-2-9-8-4-3-7-6-5`, transferring at the exact minimum of the test cycle null

Collapsed states at iterations 512 and 1024 have natural-cycle percentiles of 3.78% and 3.85%, and the collapsed update at 1024 has a percentile of 2.99%. The corresponding ideal-distance correlations remain only .08–.27. These are exploratory findings among 96 model × representation × horizon slices, without a multiple-testing correction. The concentration occurs as the collapsed trajectory degrades, does not pool into a below-5% checkpoint result, and does not reproduce in the healthy models. It should be treated as a useful collapse signature to follow up, not as evidence for the main number-helix hypothesis.

## Output-head control

The nine unit-normalized output-head rows are close to an eight-dimensional equidistant categorical arrangement. Across checkpoints, their effective rank is 7.97–7.98, the first three centroid axes contain 39.0–39.9%, and pairwise-distance coefficient of variation is only 1.4–2.0%. Natural-circle distance correlations range from -.074 to -.020 and natural-helix correlations from approximately .000 to .055. The output head is therefore much closer to a regular 8-simplex than to a natural-order ring or helix.

## Plot inspection

The [model comparison](model_comparison.png) summarizes the intrinsic and supervised results without mixing their interpretations. The [stable-state PCA atlas](stable_plain_unit_state_pca_atlas.png) and corresponding [stable-update atlas](stable_plain_unit_update_pca_atlas.png) show lobes, fans, and iteration-dependent movement rather than repeated turns around an axis. Confidence and margin are the clearest continuous gradients; true/predicted digit regions become readable, while cell-position colors mix rather than forming a repeated spatial winding.

The [combined-margin 3D figure](combined_margin_geometry_3d.png) makes the contrast explicit: the top PCA panels show the held-out geometry found without digit targets, while the lower readout panels show the circle and helix that were deliberately assigned to the digit labels. The [collapsed centroid diagnostics](collapsed_plain_centroid_diagnostics.png) show the natural-order polylines crossing the interior and the exact 20,160-cycle null. The [stable selected-cell trajectories](stable_plain_selected_cell_trajectories.png) move directly or jump between class regions; they do not share repeated winding. Lines connect log-spaced snapshots and must not be read as every recurrent step.

All final figures were opened after rendering. The 29 PNGs have readable labels and colorbars, no panel collisions, a shared digit palette, distinguishable train/test centroid paths, and an explicit cell-position key. The 3D panels use matched axis scaling so visual elongation is not an aspect-ratio artifact.

## Limits

- PCA is linear and prioritizes high-energy directions. A small nonlinear helix could be invisible to PCA; the original-space centroid tests address low-variance linear geometry but do not rule out nonlinear or state-dependent topology.
- The trajectory plots connect twelve log-spaced snapshots, so they can miss winding between snapshots.
- The final test split contains 25 puzzles. The bootstrap measures puzzle-sampling uncertainty for this split, not checkpoint-training uncertainty.
- The horizon-specific cycle scans are exploratory and correlated. Their percentiles are exact ranks among digit cycles, not multiplicity-adjusted model-level p-values.
- Only four selected checkpoints were tested. The result describes these models and this puzzle sample, not every possible Sotaku training run.

## Artifacts and reproduction

- Structured metrics and complete configuration: [`helix_metrics.json`](helix_metrics.json)
- Run log: [`analysis.log`](analysis.log)
- Analysis and statistics: [`analyze_helix.py`](analyze_helix.py), [`helix_geometry.py`](helix_geometry.py)
- Rendering: [`render_helix.py`](render_helix.py), [`finalize_artifacts.py`](finalize_artifacts.py)
- Modal entrypoint: [`modal_helix.py`](modal_helix.py)
- Tests: [`test_helix.py`](test_helix.py)
- Per model: unit-state and unit-update PCA atlases, supervised periodic atlases, a 3D comparison, centroid diagnostics, selected-cell trajectories, and a 6,000-row held-out `*_projection_samples.npz` file.

The four-checkpoint run can be launched from the repository root with:

```sh
source venv/bin/activate
modal run --detach looping/trajectory_viz/helix_tests/visual/modal_helix.py --examples-per-bucket 20 --seed 20260807
```

The local statistical tests are:

```sh
source venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=looping/trajectory_viz/helix_tests/visual/.mplconfig python -m unittest looping.trajectory_viz.helix_tests.visual.test_helix
```

The final run completed on an H200 in 114.8 seconds. Eight unit tests pass, all four NPZ samples are finite and aligned with the 25 test puzzles through iteration 1024, and all repository outputs written by this investigation are under this directory.
