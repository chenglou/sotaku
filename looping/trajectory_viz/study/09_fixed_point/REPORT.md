# Fixed-point and settling study

## Hypothesis

A healthy recurrent Sudoku model should settle toward a fixed state, while a collapsed model should fail to settle. This analysis tests raw update size, update direction, relative acceleration, normalized-state movement, distance to a late state, and finite-difference contraction of the recurrent map.

## Protocol

The analysis used the four prescribed checkpoints and 60 balanced test puzzles: 20 discovery, 20 validation, and 20 final holdout puzzles, with four puzzles from each rating bucket in every split. All metrics were specified in code before reading the final holdout. No projection was fitted.

Controls include shuffled iteration order, another puzzle's iteration-1024 state as a false endpoint, three random perturbation directions per puzzle, and a perturbation aligned with the model's own update. The same conclusions appeared in discovery, validation, and final holdout.

## Final holdout results

The raw hidden state does not approach an ordinary fixed point in any model. Median whole-board update norms at iteration 1024 were 64.2 for stable plain, 75.4 for collapsed plain, 43.0 for late-state CE, and 79.0 for the combined model. The late-state CE and combined models stayed fully solved despite flat or increasing late update norms. Forcing the raw update toward zero would therefore target a property that the healthy models do not have.

All four models eventually move in an almost perfectly constant direction: median consecutive-update cosine at iteration 1024 was at least 0.99996. Shuffling iteration order reduced mean selected-update cosine by 0.12 to 0.23, so the temporal ordering is real, but straight motion is not sufficient for health.

The collapsed checkpoint differs in how quickly its normalized state continues to drift. From iteration 896 to 1024, median normalized-state movement was 0.1027 for collapsed plain, compared with 0.0064 for stable plain, 0.0254 for late-state CE, and 0.0189 for the combined model. Median relative acceleration at iteration 1024 was 0.00849 for collapsed plain, versus 0.00103, 0.00361, and 0.00151 respectively. At iteration 768, distance to each puzzle's own iteration-1024 normalized state was 0.1948 for collapsed plain, compared with 0.0148 to 0.0569 for the three healthy checkpoints. Distances to another puzzle's endpoint were much larger, 0.82 to 1.35, confirming that the endpoint comparison measures puzzle-specific trajectory structure rather than a universal direction.

The behavioral difference matches the drift difference. On final holdout puzzles, collapsed plain fell from 90% solved at iteration 128 to 10% at iteration 1024. Stable plain, late-state CE, and the combined model were 100% solved at both horizons.

Generic local contraction did not diagnose collapse. At iteration 1024, median gain for random perturbations was below one for every checkpoint, including 0.918 for collapsed plain. Gain along the model's own update direction was approximately one for every checkpoint: 0.99998 for stable plain, 1.00011 for collapsed plain, 1.00002 for late-state CE, and 1.00003 for combined. Random directions in a 10,368-dimensional state can miss narrow unstable directions, while differences this close to one are sensitive to finite-difference scale and numerical precision.

## Verdict

The healthy models exhibit **projective settling**, not convergence to a raw hidden-state fixed point. Their hidden states keep translating, but the direction of the normalized state changes slowly and the decoded solution remains stable. The collapsed model also moves smoothly and almost straight; it simply keeps changing its normalized state several times faster, eventually changing the answer.

This result argues against update norm, visual smoothness, or average random-direction contraction as root health metrics. Late normalized-state velocity, relative acceleration, and output stability are better grounded diagnostics. These measurements identify the failure but do not yet establish whether penalizing normalized-state movement during training would fix it.

## Limitations

- The iteration-1024 state is an observed endpoint, not a proven asymptote, and the analysis does not run beyond iteration 1024.
- Finite differences sample random directions and the trajectory direction; they do not estimate the largest Jacobian singular value.
- Each model condition is represented by one checkpoint. The final holdout has 20 puzzles, although discovery and validation reproduce the main ordering.
- Normalizing the state intentionally removes magnitude. The result describes representational direction and output stability, not convergence of the complete hidden vector.

## Artifacts

- `fixed_point_metrics.json`: full split-level metrics and configuration.
- `fixed_point_summary.png`: update norm, direction persistence, endpoint distance, and update-direction gain.
- `settling_and_accuracy.png`: relative acceleration, movement over 128 iterations, and solved fraction.
- `controls.png`: shuffled-time and shuffled-endpoint controls.
- `analyze_fixed_point.py`, `modal_fixed_point.py`, and `test_fixed_point.py`: reproducible analysis and focused tests.
