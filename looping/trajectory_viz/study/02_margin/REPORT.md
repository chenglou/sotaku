# Study 02: Correct-Answer Margin

## Hypothesis

The correct-answer margin is the correct digit's score minus the highest incorrect digit's score. The hypothesis is that a hidden-state direction tracks this margin, and that models retaining correct answers preserve a positive margin as they continue iterating.

## Protocol

Run `margin_boundary_v1_20260811` sampled 60 test puzzles with seed 20260811 and split whole puzzles before fitting: 20 discovery, 20 validation, and 20 final puzzles, with four puzzles from each of the five rating buckets in every split. It used the four checkpoints required by the study protocol. Discovery fitted predictors and projections. Validation selected residual-PCA rank 16 from 1, 2, 4, 8, and 16. The primary analysis used the failing original checkpoint at iteration 128, with loss of cell correctness at iteration 1024 as the outcome. Only cells that remained correct over the predefined 64-iteration interval were eligible. The final split was then evaluated once.

The cell-level comparison included the current correct-answer logit margin, an unconstrained margin-history classifier, an ordered margin-history classifier, full recent logit history, raw hidden state, and a PCA projection after removing the output head's eight digit-contrast directions. For that removal, center the nine output-weight rows and project the state perpendicular to their row space; this does not remove every representation of digit identity. The board-level analysis used the minimum margin, tenth-percentile margin, and weakest recent margin slope among eligible blank cells. All masks were defined from originally blank cells and the predeclared correctness guard rather than from the final outcome.

## Held-out results

Within each checkpoint, one fitted hidden-state axis decoded correct-answer margin strongly on final puzzles: held-out R² was 0.960 for stable plain, 0.987 for collapsed plain, 0.947 for later-iteration training, and 0.981 for combined margin. The shuffled-margin control had mean R² -0.043 with a 95% range of [-0.137, 0.083], and the best of 32 random one-dimensional projections reached 0.820. This supports a checkpoint-local ordered margin coordinate.

On the primary collapsed-checkpoint test, 222 of 1,031 eligible final cells were wrong at iteration 1024. Current margin predicted those losses with AUC 0.795 and average precision 0.407. The unconstrained margin-history probe reached AUC 0.828, while the ordered probe reached 0.835 and average precision 0.485. The ordered model's top-risk decile contained 2.77 times the baseline event rate, compared with 2.50 times for the unconstrained model. The puzzle-cluster bootstrap for ordered history minus current margin had median AUC gain 0.039, but its 95% interval [-0.010, 0.127] crossed zero. Across observation times the ordered and unconstrained probes traded places, so the final data do not establish a general advantage for the ordered restriction.

Raw hidden state reached AUC 0.885. A predictor using 16 PCA components after that removal reached AUC 0.879 and average precision 0.584, 0.044 AUC above ordered margin history with a puzzle-cluster 95% interval [0.012, 0.077]. That increment is predictive, but it is not evidence for a special low-rank geometry: matched-rank random subspaces had mean AUC 0.875, a 95% range [0.870, 0.880], and a maximum of 0.882. The observed PCA score lies inside that control range.

The temporal result is clearer. Among final collapsed-checkpoint cells, the median margin of cells that later failed fell from 65.9 at iteration 128 to -2.15 at iteration 1024, while surviving cells rose from 96.2 to 129.0. Stable-plain survivors rose from 138.8 to 799.5; later-iteration training survivors rose from 15.1 to 17.1; combined-margin survivors stayed positive and nearly flat, 13.2 to 12.9. At board level, 17 of 18 eligible collapsed-checkpoint final puzzles collapsed, compared with 0 of 20 eligible final puzzles for each healthy checkpoint. Absolute margin and short-term slope were not universal health measures: the healthy late-state and combined checkpoints had much smaller margins and slightly negative recent slopes at iteration 128.

## Controls and transfer

Shuffling margin labels reduced the primary early-warning AUC to 0.500 on average, with a 95% range [0.470, 0.541]. Shuffling history iteration order retained AUC 0.828 on average, with a 95% range [0.824, 0.831], versus 0.835 for the ordered history; temporal order therefore added only a small amount beyond the unordered history. The matched-rank random-subspace result above prevents interpreting the residual-PCA predictor as a privileged geometry.

The scalar margin axes did not transfer across checkpoints. Applying a source checkpoint's axis unchanged to another checkpoint usually produced negative R²; the only positive off-diagonal result was later-iteration training to combined margin at 0.046. Outcome-predictor transfer was also attempted unchanged, but the healthy target final splits had no loss events, so their AUCs are undefined. The transfer evidence refutes a universal axis and shows that margin scale and orientation are checkpoint-specific.

## Limitations

The final split contains only 20 puzzles, so puzzle-level uncertainty is substantial. Only the collapsed checkpoint supplied enough future-loss events to fit and score the primary categorical outcome probes; absence of events in healthy checkpoints is meaningful behaviorally but prevents an AUC comparison. Correct-answer margin is computed from the model's own linear output head, so within-checkpoint decodability is partly expected. The board-level results are descriptive summaries rather than a separately powered board classifier. There were 32 repetitions for each permutation or random-subspace control, which limits tail resolution.

## Verdict

**Partially supported.** Hidden states predict the margin within each checkpoint, and margins fall below zero when cells become incorrect. The fitted direction does not transfer across checkpoints. Ordered margin history has no established advantage over an unrestricted history predictor, and random subspaces match the extra predictive value of residual PCA. The study does not identify a universal or uniquely low-dimensional margin direction.

## Artifacts

The authoritative machine-readable output is in [`margin_boundary_v1_20260811/metrics.json`](margin_boundary_v1_20260811/metrics.json), with compact tables in [`observation_metrics.csv`](margin_boundary_v1_20260811/observation_metrics.csv) and [`primary_predictions.csv`](margin_boundary_v1_20260811/primary_predictions.csv). The inspectable summary is [`index.html`](margin_boundary_v1_20260811/index.html); its four PNGs were visually checked at full resolution.
