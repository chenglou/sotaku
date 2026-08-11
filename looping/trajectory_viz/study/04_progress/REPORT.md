# Progress coordinates

## Hypothesis

Sotaku's board-level hidden state contains a low-dimensional coordinate that tracks solution progress: solved cells, remaining wrong cells, solve latency, and how long the board has remained solved.

## Method and controls

The run used all four reference checkpoints and 60 balanced test puzzles, split before fitting into 20 discovery, 20 validation, and 20 final puzzles. Model selection compared five board summaries and five ridge strengths on validation only. The selected pooled coordinate used the unit-normalized mean state over originally blank cells. Controls shuffled iteration labels during fitting, shuffled puzzle identities, shuffled final iteration order, and compared 64 random orthonormal directions. All reported target correlations use final held-out puzzles.

## Held-out result

One pooled coordinate tracked the three predefined progress targets with mean within-puzzle Spearman 0.811 across checkpoints. For correct-cell fraction, mean within-puzzle correlations were 0.735 for stable plain, 0.797 for collapsed plain, 0.734 for late-state CE, and 0.736 for combined margin. Correlation with remaining wrong cells had the opposite sign. Correlation with stable solved duration was especially high: 0.988, 0.879, 0.935, and 0.974 respectively.

The coordinate was mostly monotonic for the healthy checkpoints but not perfectly so. The fraction of adjacent sampled iterations that did not decrease was 0.958 for stable plain, 0.719 for late-state CE, 0.853 for combined margin, and 0.733 for collapsed plain. The collapsed checkpoint's coordinate also had a much weaker direct relation to iteration, consistent with its later reversal.

Iteration shuffling destroyed the result: the per-checkpoint shuffled-fit medians ranged from -0.015 to 0.162, versus observed values near 0.821. Random directions were weaker, with 95th percentiles from 0.694 to 0.748. Shuffling puzzle identities reduced the score only slightly for healthy checkpoints. That control matters: most puzzles follow a similar solve-time curve, so the coordinate is largely a shared time/progress axis rather than a precise puzzle-specific counter of wrong cells.

## Verdict

There is a robust, transferable board-level progress coordinate. It tracks meaningful solving behavior and exposes the collapsed checkpoint's loss of monotonic progress. The evidence does not establish a dedicated internal solved-cell counter: progress, iteration, and typical solve latency remain strongly correlated, and the puzzle-shuffle control shows that much of the signal is generic across puzzles.

## Artifacts

`metrics.json` contains the complete held-out metrics and controls. `progress_trajectories.png`, `heldout_target_tracking.png`, `checkpoint_transfer.png`, and `controls.png` show the selected coordinate and null comparisons. The five focused tests in `test_progress.py` pass.
