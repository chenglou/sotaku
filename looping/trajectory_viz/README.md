# Recurrent trajectory visualizations

These exploratory artifacts ask whether Sotaku's recurrent state follows a simple geometric path analogous to the number helices observed in language models.

## Main result

The recurrent motion is smooth and compact, but these tests do not find a shared helix. Whole-board trajectories usually look like puzzle-specific rays or gently bending paths. Digit identity is linearly accessible, but the numeric order `1, 2, ..., 9` is no more privileged than shuffled digit orders. A supervised cyclic projection can draw a ring even when the requested order is not naturally present, so the ring plots should not be treated as evidence by themselves.

The clearest difference between healthy and collapsed checkpoints is late behavior. Healthy update magnitude continues settling and per-cell directions remain spatially coherent. In the collapsed checkpoint, update magnitude begins increasing again after iteration 256 and spatial coherence falls from 0.91 at iteration 16 to 0.78 at iteration 1024.

The follow-up [twelve-part held-out study](study/README.md) tested semantic coordinates, temporal modes, settling, causal interventions, and adversarial projection controls. It found strong representations of current Sudoku state and solve progress, but rejected a universal helix, loop, arc, raw fixed point, solvedness direction, or transferable collapse-warning direction. Healthy models settle mainly in normalized direction and output behavior, not by becoming stationary in raw hidden-state space.

## Artifacts

- [`global_pca/stable_plain_global_pca.png`](global_pca/stable_plain_global_pca.png): stable whole-board trajectories in one PCA basis
- [`global_pca/collapsed_plain_global_pca.png`](global_pca/collapsed_plain_global_pca.png): collapsed trajectories bend and separate more at late iterations
- [`controls/stable_plain_local_normalized_states_3d.png`](controls/stable_plain_local_normalized_states_3d.png): normalization and projection controls
- [`controls/REPORT.md`](controls/REPORT.md): why attractive arcs are not stable across projections
- [`spatial/artifacts_downloaded/spatial/index.html`](spatial/artifacts_downloaded/spatial/index.html): interactive per-cell spatial gallery
- [`spatial/artifacts_downloaded/spatial/aggregate_geometry.png`](spatial/artifacts_downloaded/spatial/aggregate_geometry.png): healthy versus collapsed update magnitude and spatial coherence
- [`helix_tests/artifacts/stable_plain_cyclic_projection.png`](helix_tests/artifacts/stable_plain_cyclic_projection.png): a supervised cyclic digit projection; useful as a warning that the probe can manufacture a ring
- [`helix_tests/artifacts/helix_results.json`](helix_tests/artifacts/helix_results.json): held-out cyclic, categorical, and shuffled-order measurements
- [`study/README.md`](study/README.md): controlled study of uncertainty, margin, constraints, progress, digit symmetry, cell roles, difficulty, temporal modes, settling, early warning, and causal axes
