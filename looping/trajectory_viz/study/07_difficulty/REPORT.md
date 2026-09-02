# Arm 07: difficulty and solve latency

## Hypothesis

Recurrent hidden-state geometry may encode either the dataset's Sudoku difficulty rating or the number of iterations the model needs to reach a solution. The study tests both claims separately because dataset rating and model solve latency are not equivalent.

## Protocol

The analysis sampled 60 test puzzles, with 12 puzzles from each of the five rating buckets. Each bucket contributed four puzzles to discovery, four to validation, and four to the untouched final split, giving 20 puzzles per split. The fixed seed was `7027`.

For each of the four checkpoints, the analysis measured normalized whole-board state displacement, normalized update direction, and five scalar geometric statistics at iterations 4, 8, 16, 32, 64, and 128. First-solve and stable-solve latency were observed through iteration 1024; an unsolved puzzle received the censored value 1025. PCA bases were fitted only on discovery puzzles. Validation selected the representation, horizon, PCA rank, and ridge penalty. Final results were then evaluated once.

Controls included a clue-count baseline, clue count plus geometry, 100 discovery-label shuffles, matched-rank random projections for PCA results, transfer across all four checkpoints, and an unconstrained five-class rating-bucket probe with its own 100 label shuffles.

## Final held-out results

Geometry did not reliably encode dataset rating across checkpoints. Final Spearman correlations were 0.53 for stable plain, 0.61 for collapsed plain, 0.26 for later-iteration training, and 0.27 for combined margin. Only the collapsed checkpoint exceeded its shuffled-label distribution (`p=0.010`), so the association did not transfer to the three healthy checkpoints. Five-class rating-bucket accuracy was 0.35–0.40 against 0.20 chance, but none passed its shuffled-label control (`p=0.069–0.119`). The clue-count baseline was also uninformative on this balanced sample (`rho=-0.12`).

Geometry tracked solve latency much more strongly in healthy models. Stable-solve latency had final Spearman correlations of 0.95 for stable plain, 0.86 for later-iteration training, and 0.86 for combined margin. Stable plain passed the label-shuffle control (`p=0.010`). Combined margin was at the threshold (`p=0.050`) and its four-component state-displacement probe beat 20 matched random projections (`p=0.048`). Later-iteration training used the same iteration-8, four-component representation and also beat random projections (`p=0.048`), although its label-shuffle result was not significant with 100 permutations (`p=0.109`). Clue count alone was much weaker (`rho=0.20–0.32`).

The collapsed checkpoint separated initial solving from reliable settling. Its iteration-64 scalar geometry correlated with first-solve latency (`rho=0.86`) but did not pass the shuffle control (`p=0.089`). Stable-solve latency fell to `rho=0.36`, had negative final R², and failed both shuffle and random-projection controls. Only 8 of 60 collapsed-checkpoint puzzles were solved at iteration 1024, compared with 57, 60, and 59 for stable plain, later-iteration training, and combined margin.

## Verdict

The supported result is that recurrent geometry contains a strong ordering of model solve progress in healthy checkpoints. The result transfers across the three healthy models and is not explained by puzzle clue count. The compact state-displacement direction at iteration 8 is sufficient for both late-state-trained checkpoints.

The study does not support a general geometric representation of the dataset's difficulty rating. One checkpoint produced a held-out rating association, but the effect did not replicate across checkpoints or in the categorical control.

## Limitations

The final split contains only 20 puzzles, and 100 permutations give coarse empirical probabilities. Solve-latency probes at iteration 64 can describe progress already made by that point; they are not necessarily early forecasts. The high rank correlations also coexist with modest absolute calibration, especially for the slowest or censored puzzles. A larger confirmatory run should freeze the iteration-8 four-component state-displacement probe before sampling new puzzles and distinguish prediction of future solve time from measurement of current progress.

## Artifacts

- `artifacts/protocol_v2/difficulty_metrics.json`: complete configuration, split indices, selected probes, final predictions, and controls.
- `artifacts/protocol_v2/heldout_predictions.png`: final actual-versus-predicted rating and stable-solve latency.
- `artifacts/protocol_v2/summary_controls.png`: checkpoint comparison, rating-bucket accuracy, and clue-count baseline.
- `analyze_difficulty.py`: analysis and plotting code.
- `modal_difficulty.py`: detached Modal wrapper.
- `test_difficulty.py`: focused deterministic tests.
