# Study 11: Causal Interventions

## Hypothesis

Small interventions along a semantic hidden-state direction should change the corresponding quantity immediately and should alter later solving or collapse more than an equally small random-direction pulse. I tested two axes:

- Correct-digit score direction (`answer_evidence` in artifacts): for each true digit, the output-head weight for that digit minus the mean of the other eight weights, normalized to unit length. Moving along this direction changes the correct digit's logit relative to the mean of the alternatives. It uses the answer key and is constructed separately for each checkpoint; it is a positive control, not a usable solving method.
- Linear accuracy probe (`solvedness_progress`): fitted by ridge regression separately for each checkpoint, predicting the fraction of blank cells currently correct from their mean hidden state. Iteration means were removed before fitting. Predicting accuracy does not establish that changing the state along the fitted direction will improve accuracy.

## Design and controls

The canonical `sudoku-extreme` test set was sampled once with seed 20260811. Discovery, validation, and final holdout each contain 20 puzzles, with four from every rating bucket. The split was fixed before fitting. Discovery fitted the progress axes; validation selected ridge λ=0.1 from `[0.01, 0.1, 1, 10, 100]` by mean partial correlation across the four checkpoints. `discovery_validation.json` and `frozen_axes.pt` were written before the final indices were materialized for evaluation.

The final pass used the four checkpoints defined in [the protocol](../PROTOCOL.md#models). Each pulse was applied once after iteration 16 or 512 and followed to iteration 128 or 1024. Signed doses were ±0.125, ±0.25, and ±0.5 times that puzzle's natural one-step update norm. Every dose had eight orthogonal random directions with exactly the same norm. The current-accuracy predictor also had 64 within-iteration label-shuffle directions; the correct-digit direction had 64 matched random digit-conditioned directions. Checkpoint fingerprints were verified after the directions were frozen.

## Held-out results

Both directions predict their intended quantity on new puzzles. Correct-digit partial correlations were 0.940–0.996, versus random-direction 95th percentiles of 0.172–0.653. Current-accuracy correlations were 0.746–0.844, versus label-shuffle 95th percentiles of 0.304–0.653. Every fitted predictor exceeded all 64 nulls (empirical one-sided p=1/65), so the accuracy direction is a genuine correlate rather than a fitting artifact.

The causal results separate the two axes:

| checkpoint | baseline solved at 128 / 1024 | answer probe r | progress probe r | +0.5 answer pulse: immediate margin at iter. 16 / 512 | +0.5 answer pulse: solved change at 128 / 1024 |
|---|---:|---:|---:|---:|---:|
| stable plain | 100% / 100% | 0.940 | 0.811 | +4.751 / +2.100 | 0 / 0 pp |
| collapsed plain | 90% / 5% | 0.983 | 0.746 | +3.264 / +1.492 | +10 / 0 pp |
| later-iteration training | 100% / 100% | 0.964 | 0.844 | +1.937 / +0.588 | 0 / 0 pp |
| combined margin | 100% / 100% | 0.996 | 0.830 | +2.185 / +1.088 | 0 / 0 pp |

Moving along the correct-digit direction produced the predicted signed, monotone immediate margin response at both pulse times in all four checkpoints. At dose +0.5, every bootstrap interval for the immediate margin excluded zero, while random-direction p10–p90 ranges stayed close to zero. The effect also survived one recurrent step in every checkpoint. Changing the state along this answer-key-based direction therefore changes the margin, rather than merely predicting it.

Downstream solving did not transfer. The only substantial gain was on collapsed plain after the early +0.5 pulse: 18/20 to 20/20 at iteration 128, or +10 percentage points (paired bootstrap 95% CI 0 to 25; random-direction p10–p90 −1.5 to +5). A +0.25 late pulse changed collapsed-plain iteration-1024 solving from 1/20 to 2/20, but +0.5 returned to 1/20, so the late result is not a monotone dose response. The three healthy checkpoints were already 20/20 at both endpoints, leaving no room for improvement. Negative answer pulses sometimes harmed early solving, but not consistently enough to establish a transferable endpoint effect.

The fitted current-accuracy direction failed the causal test. Its +0.5 early pulse changed immediate cell accuracy by +0.27, −0.09, −0.53, and −1.67 percentage points in the failing original, combined, later-iteration training, and accurate original models. Every value lay inside its checkpoint's random-direction range, the sign did not transfer, and endpoint solved changes were zero at +0.5. Smaller signed doses produced isolated ±5-point changes on the 20-puzzle sample, but those changes were non-monotone and within matched-control ranges. Predicting current accuracy therefore did not provide a reliable way to improve it by modifying the state.

## Limitations

- Twenty final puzzles make solved accuracy move in five-point increments and give wide intervals. Three checkpoints score 100% at both endpoints, so the experiment mainly tests whether a pulse can damage their accuracy, not improve it.
- `answer_evidence` uses the ground-truth digit and directly changes output-head logits. It is a positive causal control, not a deployable inference method.
- Axes were defined independently in each checkpoint because raw hidden coordinates are not identifiable across separately trained models. Cross-checkpoint evidence here means replication of the same semantic construction and intervention protocol, not transfer of one raw 128-dimensional vector.
- There are eight downstream random directions and one puzzle sample. Their p10–p90 bands are controls, not precise null quantiles.
- The maximum pulse is half of one natural recurrent update. The findings do not cover persistent interventions, other recurrence points, or larger state edits.

## Verdict

**The answer-key-based direction changes margins; the fitted accuracy direction does not reliably improve solving.** A small pulse along the correct-digit-versus-mean-alternative direction changes correct-answer margin across all four checkpoints, and recurrence preserves part of that change for at least one step. The intervention does not reliably prevent later accuracy loss or improve eventual solving across checkpoints. The learned accuracy direction predicts held-out results but fails sign, dose, random-direction, and downstream-outcome controls. These results do not establish a state direction that can reliably control solve progress.

Artifacts: `final_metrics.json` contains puzzle-level paired effects, bootstrap intervals, and random-control distributions; `discovery_validation.json` records the split and model-selection audit; `dose_response.png`, `immediate_semantics.png`, and `heldout_probes.png` show the held-out results.
