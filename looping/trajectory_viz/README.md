# Recurrent trajectory visualizations

These exploratory artifacts ask whether Sotaku's recurrent state follows a simple geometric path analogous to the number helices observed in language models.

## Main result

The recurrent motion is smooth and compact, but these tests do not find a shared helix. Whole-board trajectories usually look like puzzle-specific rays or gently bending paths. Digit identity is linearly accessible, but the numeric order `1, 2, ..., 9` is no more privileged than shuffled digit orders. A supervised cyclic projection can draw a ring even when the requested order is not naturally present, so the ring plots should not be treated as evidence by themselves.

The clearest difference between the accurate and failing checkpoints is their later behavior. In the accurate checkpoint, update magnitude continues decreasing and per-cell update directions remain similar across the board. In the failing checkpoint, update magnitude begins increasing again after iteration 256 and the measure of that directional agreement falls from 0.91 at iteration 16 to 0.78 at iteration 1024. The plots retain the historical labels `stable_plain` and `collapsed_plain`; these describe the selected checkpoints, not guarantees about their training recipes.

The follow-up [twelve-part held-out study](study/README.md) tested which quantities the state represents, how its motion changes, and whether modifying candidate directions changes the answers. It found strong representations of current Sudoku state and solve progress, but no universal helix, loop, arc, hidden-state fixed point, direction that reliably improves solving, or transferable warning of later accuracy loss. In the accurate models, state direction changes slowly at the measured later iterations while predictions remain correct; the complete hidden state does not stop moving.

## Artifacts

- [`global_pca/stable_plain_global_pca.png`](global_pca/stable_plain_global_pca.png): stable whole-board trajectories in one PCA basis
- [`global_pca/collapsed_plain_global_pca.png`](global_pca/collapsed_plain_global_pca.png): collapsed trajectories bend and separate more at late iterations
- [`controls/stable_plain_local_normalized_states_3d.png`](controls/stable_plain_local_normalized_states_3d.png): normalization and projection controls
- [`controls/REPORT.md`](controls/REPORT.md): why attractive arcs are not stable across projections
- [`spatial/artifacts_downloaded/spatial/index.html`](spatial/artifacts_downloaded/spatial/index.html): interactive per-cell spatial gallery
- [`spatial/artifacts_downloaded/spatial/aggregate_geometry.png`](spatial/artifacts_downloaded/spatial/aggregate_geometry.png): update magnitude and agreement between cell update directions in accurate versus failing checkpoints
- [`helix_tests/artifacts/stable_plain_cyclic_projection.png`](helix_tests/artifacts/stable_plain_cyclic_projection.png): a projection fitted to place digits on a circle; the fitted circle alone does not establish the original geometry
- [`helix_tests/artifacts/helix_results.json`](helix_tests/artifacts/helix_results.json): held-out cyclic, categorical, and shuffled-order measurements
- [`study/README.md`](study/README.md): controlled study of uncertainty, margin, constraints, progress, digit symmetry, cell roles, difficulty, temporal modes, settling, early warning, and causal axes
