# Arm 11 — causal semantic axes

## Hypothesis

Small interventions along a semantic hidden-state direction should change the corresponding quantity immediately and should alter later solving or collapse more than an equally small random-direction pulse. I tested two axes:

- `answer_evidence`: for each true digit, the output-head weight for that digit minus the mean of the other eight weights. This is an oracle, checkpoint-local positive control.
- `solvedness_progress`: a checkpoint-local ridge direction from the mean blank-cell state to the fraction of blank cells currently correct. Iteration means were removed before fitting.

## Design and controls

The canonical `sudoku-extreme` test set was sampled once with seed 20260811. Discovery, validation, and final holdout each contain 20 puzzles, with four from every rating bucket. The split was fixed before fitting. Discovery fitted the progress axes; validation selected ridge λ=0.1 from `[0.01, 0.1, 1, 10, 100]` by mean partial correlation across the four checkpoints. `discovery_validation.json` and `frozen_axes.pt` were written before the final indices were materialized for evaluation.

The final pass used the protocol's stable plain, collapsed plain, standalone late-state-CE, and combined-margin checkpoints. Each pulse was applied once after iteration 16 or 512 and followed to iteration 128 or 1024. Signed doses were ±0.125, ±0.25, and ±0.5 times that puzzle's natural one-step update norm. Every dose had eight orthogonal random directions with exactly the same norm. The supervised probe also had 64 within-iteration label-shuffle axes; answer evidence had 64 matched random digit-conditioned axes. Checkpoint fingerprints were verified after the axes were frozen.

## Held-out results

Both axes decode their intended quantity on new puzzles. Answer-evidence partial correlations were 0.940–0.996, versus random-axis 95th percentiles of 0.172–0.653. Solvedness-progress correlations were 0.746–0.844, versus label-shuffle 95th percentiles of 0.304–0.653. Every real probe exceeded all 64 nulls (empirical one-sided p=1/65), so the progress axis is a genuine correlate rather than a fitting artifact.

The causal results separate the two axes:

| checkpoint | baseline solved at 128 / 1024 | answer probe r | progress probe r | +0.5 answer pulse: immediate margin at iter. 16 / 512 | +0.5 answer pulse: solved change at 128 / 1024 |
|---|---:|---:|---:|---:|---:|
| stable plain | 100% / 100% | 0.940 | 0.811 | +4.751 / +2.100 | 0 / 0 pp |
| collapsed plain | 90% / 5% | 0.983 | 0.746 | +3.264 / +1.492 | +10 / 0 pp |
| late-state CE | 100% / 100% | 0.964 | 0.844 | +1.937 / +0.588 | 0 / 0 pp |
| combined margin | 100% / 100% | 0.996 | 0.830 | +2.185 / +1.088 | 0 / 0 pp |

The answer-evidence intervention produced the predicted signed, monotone immediate margin response at both pulse times in all four checkpoints. At dose +0.5, every bootstrap interval for the immediate margin excluded zero, while random-direction p10–p90 ranges stayed close to zero. The effect also survived one recurrent step in every checkpoint. Thus answer evidence is causally readable in the current and next state, not merely correlated.

Downstream solving did not transfer. The only substantial gain was on collapsed plain after the early +0.5 pulse: 18/20 to 20/20 at iteration 128, or +10 percentage points (paired bootstrap 95% CI 0 to 25; random-direction p10–p90 −1.5 to +5). A +0.25 late pulse changed collapsed-plain iteration-1024 solving from 1/20 to 2/20, but +0.5 returned to 1/20, so the late result is not a monotone dose response. The three healthy checkpoints were already 20/20 at both endpoints, leaving no room for improvement. Negative answer pulses sometimes harmed early solving, but not consistently enough to establish a transferable endpoint effect.

The solvedness-progress axis failed the causal test. Its +0.5 early pulse changed immediate cell accuracy by +0.27, −0.09, −0.53, and −1.67 percentage points in collapsed, combined, late-state, and stable models. Every value lay inside its checkpoint's random-direction range, the sign did not transfer, and endpoint solved changes were zero at +0.5. Smaller signed doses produced isolated ±5-point changes on the 20-puzzle sample, but those changes were non-monotone and within matched-control ranges. The strong held-out probe therefore does not show that recurrence uses this direction as a progress control variable.

## Limitations

- Twenty final puzzles make solved accuracy move in five-point increments and give wide intervals. The healthy checkpoints are at ceiling at both endpoints, so the experiment mainly tests whether a pulse can damage them, not improve them.
- `answer_evidence` uses the ground-truth digit and directly changes output-head logits. It is a positive causal control, not a deployable inference method.
- Axes were defined independently in each checkpoint because raw hidden coordinates are not identifiable across separately trained models. Cross-checkpoint evidence here means replication of the same semantic construction and intervention protocol, not transfer of one raw 128-dimensional vector.
- There are eight downstream random directions and one puzzle sample. Their p10–p90 bands are controls, not precise null quantiles.
- The maximum pulse is half of one natural recurrent update. The findings do not cover persistent interventions, other recurrence points, or larger state edits.

## Verdict

**Accept a narrow result for answer evidence; reject solvedness progress as causal.** A small pulse along the true-digit output direction causally and dose-dependently changes correct-answer margin across all four checkpoints, and recurrence preserves part of that change for at least one step. The intervention does not reliably prevent collapse or improve eventual solving across checkpoints. The learned progress axis transfers as a held-out correlation but fails sign, dose, random-direction, and downstream-outcome controls. These results do not support a general low-dimensional “solvedness knob” in the recurrent state.

Artifacts: `final_metrics.json` contains puzzle-level paired effects, bootstrap intervals, and random-control distributions; `discovery_validation.json` records the split and model-selection audit; `dose_response.png`, `immediate_semantics.png`, and `heldout_probes.png` show the held-out results.
