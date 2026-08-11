# Arm 12 preregistration

This file and `acceptance_criteria.json` define the audit before the final holdout is evaluated.

## Hypotheses

The audit separates four claims that can otherwise be conflated by an attractive PCA plot:

1. Consecutive recurrent states or updates are temporally smoother than a shuffled ordering.
2. The smooth motion occupies a shared low-dimensional arc that transfers to unseen puzzles.
3. The shared motion closes into a loop.
4. The shared motion has helical structure: a stable transverse rotation plus monotonic motion on a third axis.

Temporal smoothness is intrinsic and does not require a projection. Arc, loop, and helix claims are projection-sensitive and must survive discovery-only PCA, a validation-selected plane, matched-rank random projections, shuffled time, unseen puzzles, representation changes, split repeats, and checkpoint transfer.

## Split and analysis choices

The canonical balanced test loader supplies twelve puzzles from each of five rating buckets using seed `2026081112`. A second deterministic, rating-stratified shuffle assigns four puzzles per bucket to discovery, four to validation, and four to final holdout. No cell-level split is allowed. Discovery fits each global PCA basis. Validation chooses only the transverse plane among the three reported axes. The final holdout is evaluated once after these choices and this preregistration are frozen.

The snapshots are iterations 0, 32, ..., 1024. The four predefined representations are raw states, board-L2-normalized states, raw one-step updates, and board-L2-normalized one-step updates. State paths are anchored at iteration 0 after optional normalization. Translation is removed per trajectory before variance calculations because a path's offset is not geometry.

Global PCA reports three axes from a discovery-fitted rank-32 basis. The projection control draws 128 random orthonormal three-dimensional subspaces inside the same discovery-fitted rank-32 span. Each random projection receives the same validation-only plane selection as PCA. Full-trajectory per-puzzle PCA is deliberately circular and appears only as an adversarial control showing how much visual structure a local fit can manufacture.

## Metrics and controls

Temporal continuity is the log ratio between a typical all-pairs distance and the mean consecutive distance. The arc score combines tangent smoothness, end-to-end efficiency, and nonzero chord deviation. Loop and helix scores use unwrapped angular linearity, net turns, radius consistency, transverse closure for loops, and absolute axial Spearman correlation for helices. Scores are aggregated by the median puzzle so puzzles with more blank cells cannot dominate.

Ordered-minus-shuffled effects use 499 independent within-puzzle time permutations. Confidence intervals use 1,000 whole-puzzle bootstrap samples. Time-shuffle p-values are adjusted within each claim family. Split robustness uses eight rating-stratified discovery/validation reassignments inside the 40-puzzle development set. Checkpoint transfer applies the stable-plain discovery basis and its stable-plain validation-selected plane to the other checkpoints without refitting.

Synthetic random walks, smooth Gaussian processes, and mixtures of monotone decays are generated without Sudoku information. Local PCA and best-of-many projection selection are then allowed to optimize their appearance. These examples are demonstrations of selection bias, not a reference distribution for accepting a real claim.

## Acceptance

The exact thresholds are machine-readable in `acceptance_criteria.json`. In brief, an individual result needs adjusted time-shuffle `q <= 0.05`, an ordered-minus-shuffle effect of at least 0.10, and a positive 95% puzzle-bootstrap lower bound. Projection-sensitive claims additionally need at least the 95th percentile among matched-rank random projections. A shared claim must pass in at least three of four representations for at least two checkpoints, pass at least 75% of development split repeats, and transfer from the stable-plain basis to at least two other checkpoints. Arc, loop, and helix claims also have the absolute geometric thresholds recorded in the JSON file.

An unconstrained categorical probe is not applicable: this arm does not claim that digit labels form an ordered or cyclic representation.
