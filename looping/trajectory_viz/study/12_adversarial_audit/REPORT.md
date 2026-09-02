# Arm 12 — adversarial audit

## Hypothesis

An attractive recurrent-state PCA path can support four increasingly strong claims: consecutive states are temporally smooth, the motion follows a shared low-dimensional arc, the arc closes into a loop, or the motion forms a helix. This audit asks which claims remain after projection choice, normalization, time order, puzzle split, and checkpoint transfer are controlled. Visual appeal is never an acceptance criterion.

## Design and controls

The audit used the four checkpoints from `looping/eval_trajectory_geometry.py`, defined in [the protocol](../PROTOCOL.md#models): the accurate and failing original checkpoints, later-iteration training alone, and the combined recipe that adds a second training window and margin penalty at step 39K. Checkpoint SHA-256 hashes are recorded in `robustness_metrics.json`.

The canonical balanced test sample was split before fitting. Discovery, validation, and final holdout each contain 20 whole puzzles, with four puzzles from each of the five rating buckets. Discovery fit each rank-32 PCA basis, validation selected a plane among the first three reported axes, and the final holdout was evaluated once. `split_manifest.json` records the split and puzzle hashes without exposing puzzle text.

The four predefined representations were raw states, board-L2-normalized states, raw one-step updates, and board-L2-normalized one-step updates at iterations 0, 32, ..., 1024. Projection-sensitive tests compared global discovery PCA with circular full-trajectory per-puzzle PCA, 128 random orthonormal rank-3 subspaces inside the same discovery rank-32 span, 499 within-puzzle time shuffles, 1,000 whole-puzzle bootstraps, eight development resplits, and transfer of the stable checkpoint's frozen basis to the other three checkpoints. Float64, scale-stabilized dual PCA avoided overflow in the raw-state covariance; all saved projections and JSON values were checked for finiteness.

Synthetic nulls contained no Sudoku information. Favorable local PCA and post hoc axis selection raised the median arc score from 0.56 to 0.72, loop score from 0.39 to 0.77, and helix score from 0.16 to 0.52. The selected examples reached 0.76, 0.88, and 0.74 respectively. These nulls demonstrate that a convincing path can be manufactured by projection selection.

## Final held-out results

| Claim | Final evidence | Preregistered verdict |
|---|---|---|
| Temporal continuity | All 16 checkpoint × representation tests passed. Ordered-minus-shuffled effects ranged from 0.73 to 2.43; every adjusted `q` was 0.002 and every puzzle-bootstrap lower bound was positive. | **Pass** |
| Shared arc | Ordered PCA paths beat shuffled time in all 16 tests, but only collapsed-plain normalized states passed the full individual criteria. No checkpoint passed in three representations, and the stable basis transferred to zero checkpoints under the matched-rank control. | **Fail** |
| Shared loop | Ordered-minus-shuffled effects ranged from -0.052 to 0.013 and every adjusted `q` was 1.0. No development resplit or final representation passed. | **Fail** |
| Shared helix | Some ordered paths beat shuffled time, especially raw states, but no test met the absolute helix geometry and projection-control thresholds. No checkpoint passed in any representation and the stable basis transferred to zero checkpoints. | **Fail** |

The arc failure is informative. On normalized states, own-basis arc scores were 0.59, 0.79, 0.78, and 0.80 for stable, collapsed, late-state, and combined-margin checkpoints. Only collapsed normalized states exceeded both the rank-3 variance threshold (0.258) and the 95th percentile of matched-rank projections (percentile 0.953). Stable-basis transfer retained visually large arc scores of 0.62–0.72 on the other checkpoints, but those scores fell between the 1st and 43rd percentiles of the matched-rank controls. The arcs therefore reflect smooth motion plus favorable axes, not a distinguished shared three-dimensional geometry.

Loop evidence was absent. Final loop scores were at most 0.045, and temporal ordering never improved the aggregate loop score reliably. Helix-like time ordering was measurable in several projections, but the strongest own-basis final helix score was 0.257, below the preregistered 0.35 threshold. The relevant paths also failed one or more of radius consistency, net turns, axial monotonicity, and matched-rank projection percentile.

As a descriptive check, all 20 final puzzles were solved at iteration 1024 by stable plain, later-iteration training, and combined margin; collapsed plain solved none. The geometry verdict does not use this small-sample accuracy result.

## Limitations

The study uses the protocol minimum of 20 puzzles per split and one trained checkpoint per condition, so it does not measure training-seed variability. Sampling every 32 iterations emphasizes long-horizon motion and can miss a short early cycle. The audit tests whole-board flattened states and updates; a localized cell, constraint, or layer trajectory could have different geometry. The random controls are matched inside each discovery rank-32 span rather than across every possible ambient-space projection. Per-puzzle PCA sees the full path and is intentionally circular; it is included only to quantify how much a favorable local fit can improve appearance. The analysis makes no ordered or cyclic digit-identity claim, so the categorical-probe control in the general protocol is not applicable.

## Verdict

The skeptical conclusion is narrow: recurrent trajectories are strongly and consistently smooth in time. The evidence does not support a shared arc, loop, or helix. Attractive arcs survive shuffled time because the underlying dynamics are smooth, but they do not survive the combination of matched-rank projection controls, representation robustness, and checkpoint transfer. The loop claim is directly contradicted by the held-out temporal control.

## Artifacts

- `robustness_metrics.json`: single-pass final metrics, controls, checkpoint hashes, and machine-readable verdicts
- `development_metrics.json`: validation results, development resplits, and synthetic-null selection measurements
- `frozen_projection_choices.npz` and `frozen_choices.json`: discovery-fitted bases and validation-only choices used by the final pass
- `adversarial_gallery.png`: deceptive synthetic nulls beside honest global and circular per-puzzle real projections
- `robustness_summary.png`: held-out effects and acceptance markers for all checkpoint × representation tests
- `checkpoint_transfer.png`: own-basis versus stable-basis normalized-state scores
- `index.html`: inspectable artifact index
- `audit_core.py`, `run_audit.py`, and `test_audit_core.py`: analysis code and focused tests

All three PNG files were inspected at full resolution. Titles, legends, time colors, pass/fail markers, and checkpoint labels are legible, and no clipping changes the interpretation.
