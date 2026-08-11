# Projection-control review

This analysis asks whether attractive curves in recurrent-state plots are stable properties of the model or artifacts of choosing a favorable projection. It uses the locally available stable baseline, five balanced test puzzles, iterations 0–1024 sampled every four iterations, and four representations: raw states, unit-normalized states, raw one-step updates, and unit-normalized updates.

## Conclusion

The model follows smooth, low-dimensional trajectories, but these results do not support a shared helix-like coordinate system.

Raw state trajectories are nearly one-dimensional: on held-out puzzles, three local PCA components explain 99.997% of variance, and path efficiency is 0.997 locally, 1.000 in a PCA basis fitted on other puzzles, and 0.986 under random projection. Most of this strikingly clean shape is accumulated state magnitude. After unit-normalizing each state, local 3D PCA explains 94.6% and median path efficiency falls to 0.50 locally, 0.93 in the cross-puzzle basis, and 0.41 under random projection.

Updates show genuine smooth temporal organization without a stable geometric shape. Held-out raw-update path efficiency is 0.43 locally, 0.42 in the cross-puzzle basis, and 0.35 under random projection. Unit-normalized updates fall to 0.32, 0.31, and 0.20. Consecutive projected updates remain aligned: median turn cosine is 0.94 for normalized updates in the cross-puzzle PCA basis, versus -0.36 after shuffling time. The ordered motion is real; the particular arcs, hooks, and occasional loops depend strongly on puzzle and projection.

Visual inspection of both 2D and 3D figures found no repeated helix across puzzles or projection methods. Local PCA often produces the cleanest curve because it is fitted to that exact trajectory. Random projections preserve temporal smoothness but substantially change the visible shape. Shuffling time leaves the same point cloud while destroying the smooth path, confirming that point-cloud appearance alone is not enough.

## Limits

Only the stable baseline checkpoint was available locally. Modal execution was not authorized, so this pass does not compare the collapsed, late-state CE, and combined checkpoints. Five puzzles are enough to reject an obvious universal helix but not enough to rule out more specific geometry, such as a representation that appears only in certain cells, digit features, or solving phases.

The next justified analysis would search for coordinates tied to interpretable variables, e.g. candidate entropy, solved-cell count, or digit identity, and test those coordinates on held-out puzzles. Looking for a visually pleasing curve without an independently specified variable is too unconstrained.

## Artifacts

- `projection_controls.json`: per-puzzle metrics and complete configuration.
- `stable_plain_local_puzzle_0.png` through `stable_plain_local_puzzle_3.png`: local PCA, cross-puzzle PCA, random projection, and shuffled-time controls for all four representations.
- `stable_plain_local_normalized_states_3d.png`: 3D local, cross-puzzle, and random projections.
- `analyze_projection_controls.py`: reproducible analysis.
- `modal_projection_controls.py`: isolated runner for the full checkpoint comparison when remote execution is explicitly authorized.
