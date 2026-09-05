# Cell roles

## Hypothesis

Recurrent states should distinguish clues from blanks, acquire spatial roles, and represent input-derived candidate-set size. A control that subtracts the iteration-0 input-symbol state tests whether these signals are merely persistent input encoding.

## Method and controls

The run used all four reference checkpoints and 60 balanced puzzles, split into 20 discovery, 20 validation, and 20 final puzzles before fitting. Probes were evaluated at iterations 0, 1, 4, 16, 128, 512, and 1024 on raw state, unit state, input-symbol residual, and unit input-symbol residual. Labels covered clue versus blank, row, column, box, categorical candidate size, and ordinal candidate size. Ridge strength was selected on validation. Controls included within-puzzle label shuffles, matched random subspaces, temporal-order permutations, and the explicit input-residual representation.

## Held-out result

Iteration 0 contains only input-symbol information. Clue versus blank is decoded perfectly, while row, column, box, and candidate-size probes are at chance. Subtracting the discovery-set mean state for each exact input symbol makes the iteration-0 residual zero on held-out cells by construction.

Recurrent iterations add information not present in that input encoding. At iteration 16, ordinal candidate-size within-puzzle Spearman was 0.593 for stable plain, 0.639 for collapsed plain, 0.695 for later-iteration training, and 0.682 for combined margin. These values were unchanged by subtracting the input-symbol mean, so the signal is recurrently computed rather than copied from the initial cell token. Candidate-size information generally weakened after puzzles were solved, although combined margin retained a correlation of 0.328 at iteration 1024.

Spatial roles also emerged recurrently. Stable plain's final balanced row/column/box accuracies at iteration 128 were 0.789/0.633/0.413, from chance 0.111, and were unchanged by input residualization. The collapsed checkpoint's spatial decoding fell sharply by iteration 1024 alongside puzzle accuracy, whereas stable plain retained strong row and column decoding. Later-iteration training and combined margin used less linearly accessible absolute-position information at deep iterations, despite remaining accurate solvers.

Raw clue-versus-blank accuracy mostly measured persistent input identity: at iteration 16 it was approximately 1.0 for every checkpoint, but after input residualization balanced accuracy fell near 0.51-0.56. Any residual clue-role information varied by checkpoint. Combined margin retained the strongest residual at iteration 1024, with balanced accuracy 0.701; stable plain was at chance by then.

## Verdict

The state acquires candidate-set size and cell-position information during recurrence, with candidate size clearest early in solving. Clue-versus-blank information mostly comes from the original input embedding; position and candidate-size information remain after that input contribution is subtracted. Which features remain readable after solving differs substantially between checkpoints.

## Limitations

Candidate size is calculated from the original Sudoku constraints, not from a dynamically updated candidate set. Linear probe accessibility does not prove that the model causally uses a feature. Several temporal-order tests are exploratory across many task and representation combinations; the machine-readable metrics should be consulted rather than treating isolated nominal p-values as discoveries.

## Artifacts

`artifacts/cell_roles_v1/metrics.json` contains all final probes and controls. The same directory contains four inspected PNG summaries and `index.html`. The nine focused tests in `test_analysis.py` pass.
