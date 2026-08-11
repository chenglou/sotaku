# Spatial trajectory visualization

This pass compares the stable plain checkpoint with a plain checkpoint that collapses at long horizons. It uses ten balanced test puzzles and a feature PCA basis fitted jointly to both models, so colors have the same meaning across the comparison.

## What the plots show

- Both models have smooth recurrent motion. Smoothness and low-dimensional projections therefore do not distinguish a healthy model by themselves.
- The update magnitude falls substantially with iteration in both models. The collapsed model reaches its minimum around iteration 256, then grows again from 4.64 at iteration 256 to 5.44 at iteration 1024. The healthy model continues approaching a plateau, from 7.43 to 6.65 over the same range.
- The healthy model's projected cell directions become strongly synchronized after iteration 16 and retain similar spatial coherence through iteration 1024: 0.94 at iteration 16 and 0.92 at iteration 1024. The collapsed model falls from 0.91 to 0.78.
- Individual boards do not show a clear wave moving through rows, columns, or boxes. Late healthy updates instead look like a stable, puzzle-specific spatial pattern whose cells move in closely related feature directions.
- The projected trajectories look like spokes or gently bending rays, not repeated helices. The same broad shape appears across puzzles, but puzzle-specific cell coefficients determine the board pattern.

The phase heatmaps use the angle in the first two shared PCA directions. This angle is useful for comparing the two checkpoints, but it has not been tied to a semantic variable such as digit identity, candidate entropy, or solved-cell count. It should not be interpreted as a discovered phase coordinate yet.

## Artifacts

- `artifacts_downloaded/spatial/index.html`: self-contained gallery.
- `artifacts_downloaded/spatial/aggregate_geometry.png`: update magnitude, spatial coherence, and projected phase change.
- `artifacts_downloaded/spatial/puzzle_*_spatial.png`: per-cell update magnitude and projected phase over iteration.
- `artifacts_downloaded/spatial/metrics.json`: plotted values and checkpoint paths.

The next grounded test is to regress the shared coordinates against independently meaningful quantities, especially each cell's answer digit, current prediction entropy, correctness, and row/column/box conflict counts. A helix-like interpretation would only become convincing if a coordinate predicts one of those quantities on held-out puzzles.
