# Digit symmetry

## Hypothesis

Digit identity should occupy a compact categorical subspace, while Sudoku's arbitrary digit names should not privilege the numeric order 1 through 9 or a particular cyclic arrangement.

## Method and controls

The run used all four reference checkpoints and 60 balanced puzzles, split into 20 discovery, 20 validation, and 20 final puzzles before fitting. Analysis used cell-centered, unit-normalized hidden-state directions from originally blank cells at iterations 16, 128, 512, and 1024. It compared an unrestricted categorical decoder, an eight-dimensional digit-category subspace, ordinal and cyclic digit codes, 199 shuffled digit orders, 199 label shuffles, 99 matched random subspaces, and transfer across puzzle splits, iterations, and checkpoints.

## Held-out result

Digit category is strongly accessible. At iteration 128, full hidden-state digit decoding on final puzzles reached 1.000 for stable plain, 0.966 for collapsed plain, 0.961 for later-iteration training, and 0.932 for combined margin. The discovery-fitted rank-eight category subspace retained 1.000, 0.953, 0.958, and 0.925 respectively. Label-shuffled decoding was near chance.

The natural numeric order was not special. For stable plain at iteration 128, the natural cyclic code explained 0.227 of categorical effect, while the validation-selected shuffled order explained 0.319 on final puzzles. At iteration 1024 the comparison was 0.203 versus 0.374. The same pattern held for the trained deep-horizon checkpoints: later-iteration training scored 0.210 versus 0.388 at iteration 1024, and combined margin scored 0.165 versus 0.436. Ordinal 1-through-9 effects were smaller still. These results replicate the earlier helix control with a larger, preregistered split and stronger nulls.

Digit-category geometry transfers across puzzle splits, but exact geometry is not universal across checkpoints. Relative digit-distance correlations between stable plain and other checkpoints were high at iteration 16, from 0.861 to 0.889 on final puzzles, then generally weakened at deeper iterations. The best cross-checkpoint digit relabeling was never the identity permutation. Models therefore preserve categorical digit information while choosing different internal arrangements of the nine interchangeable symbols.

## Verdict

Sotaku has a compact digit-category subspace, not a number helix. Digit identities are easy to decode and broadly reusable across puzzles, but numeric adjacency and the cycle 1 to 9 are not privileged. The changing geometry across checkpoints is consistent with Sudoku's digit-permutation symmetry.

## Artifacts

`metrics.json` contains all final probes, permutation controls, and alignment measures. `decoder_transfer.png`, `digit_distance_matrices.png`, `geometry_alignment.png`, and `order_controls.png` visualize the held-out results. The six focused tests in `test_core.py` pass.
