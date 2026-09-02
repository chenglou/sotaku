# Arm 10: early warning

## Hypothesis

Intrinsic recurrent geometry visible by iteration 128 predicts whether a puzzle that is solved at iteration 128 becomes unsolved by iteration 1024. The intended claim is stronger than distinguishing known checkpoints: geometry must add information beyond puzzle difficulty, target-free output confidence, and checkpoint identity, and the warning must transfer to a held-out checkpoint.

## Design

The sample contains 60 canonical test puzzles selected with random seed 20260811 and balanced across the five rating buckets. Puzzles were assigned before model inference to discovery, validation, and final splits of 20 puzzles each, with four puzzles from each rating bucket in every split. The same whole-puzzle split was used for the four checkpoints defined in [the protocol](../PROTOCOL.md#models): the accurate and failing original checkpoints, later-iteration training alone, and the combined recipe that adds a second training window and margin penalty at step 39K.

A row is eligible when every originally blank cell is correct at iteration 128. Its collapse label is 1 only when at least one originally blank cell is wrong at iteration 1024. The feature extractor does not receive targets or late states. Its fixed geometry features use states at iterations 64, 80, 96, 112, and 128 and transitions ending at those iterations, where `update[t] = hidden[t] - hidden[t - 1]`. They cover update magnitude and slope, relative update size, temporal alignment, path efficiency, turning, cell-direction coherence, norm dispersion, and state/update effective rank. No projection is fitted or used.

The fixed baselines are rating bucket plus clue fraction, target-free early output statistics, and checkpoint identity. All models use the same ridge penalty of 1.0; there is no feature selection. Coefficients and standardization are fitted on discovery puzzles. Validation puzzles fit a one-parameter logit offset for probability calibration. Final puzzles are evaluated once with AUROC, average precision, Brier score, log loss, and ECE. Bootstrap intervals resample whole puzzles and retain their paired checkpoint rows.

The primary permutation test shuffles labels within checkpoint and rating bucket, independently in each split, and repeats fitting and calibration 500 times. This preserves checkpoint prevalence and puzzle-difficulty strata. A separate control applies a fixed, label-independent within-puzzle permutation to the five early snapshots before recomputing every geometry feature. Strict leave-one-checkpoint-out models omit the target checkpoint from both discovery fitting and validation calibration.

## Held-out result

The final eligible sample has 78 checkpoint-puzzle rows and 17 collapses. All 17 collapses occur in collapsed plain: stable plain has 0/20, collapsed plain has 17/18, later-iteration training has 0/20, and combined margin has 0/20. Discovery and validation are even more completely separated by checkpoint: every eligible collapsed-plain puzzle collapses, while no eligible puzzle from the other checkpoints collapses.

This separation produces attractive but confounded pooled results. Geometry alone reaches AUROC 1.000, average precision 1.000, Brier 0.0119, and log loss 0.0852. The difficulty-plus-output baseline reaches AUROC 0.951, average precision 0.689, and log loss 0.2884. However, checkpoint identity alone reaches AUROC 0.992, average precision 0.944, and log loss 0.0646. Adding geometry to the checkpoint, difficulty, and output model changes AUROC from 0.984 to 1.000 and log loss from 0.0849 to 0.0203, but the checkpoint-conditioned permutation test is not significant: one-sided `p = 0.267` for both the 0.0164 AUROC gain and the 0.0647 log-loss improvement. The pooled full model is well calibrated on these four checkpoints (Brier 0.00082, ECE 0.0198), but that calibration mostly reflects checkpoint separation and is not evidence of transfer.

Shuffling the early iteration order retains almost all pooled separation. The checkpoint, difficulty, output, and shuffled-geometry model reaches AUROC 0.984 and log loss 0.0789, compared with 1.000 and 0.0203 for ordered geometry. Geometry without the other controls still reaches AUROC 0.974 after shuffling. Ordered time contributes an illustrative AUROC gain of only 0.0164; most of the signal is therefore a checkpoint-specific distributional difference, not evidence for a temporal precursor.

The strict transfer test fails. When collapsed plain is held out, all source discovery and validation labels are zero. Both the baseline and geometry model give AUROC 0.500 on collapsed plain, log loss 13.05, and permutation `p = 1.0`. The other three held-out checkpoints contain zero final collapse events, so their AUROCs are undefined. Their probabilities are also checkpoint-specific: adding geometry gives log loss 8.47 on held-out later-iteration training versus 0.935 for the baseline, while combined margin remains near zero risk. No early warning in this sample transfers across checkpoints.

## Controls and leakage audit

- Puzzle splits are fixed before feature extraction or model fitting. No puzzle appears in more than one split.
- Features end at iteration 128. The transition indexed 128 ends at 128 and does not inspect state 129.
- Geometry and output features are target-free. Targets are used only to define iteration-128 eligibility and the separate iteration-1024 label.
- No supervised, PCA, or other fitted projection is used, so post-128 projection leakage and random-subspace sensitivity do not apply.
- Difficulty, target-free output confidence, and checkpoint identity are explicit baselines.
- Conditional permutations preserve checkpoint and rating-bucket composition.
- A fixed within-puzzle permutation of early snapshot order is a matched control for temporal features.
- Leave-one-checkpoint-out fitting and calibration test checkpoint transfer directly.
- Probability calibration uses validation puzzles only. Final labels are used only for evaluation and permutations.

## Limitations and sample size

The protocol minimum of 20 puzzles per split is enough to expose the checkpoint confound but not enough for checkpoint-specific discrimination or calibration. No checkpoint has at least 10 collapse events and 10 non-events on the final split: the three healthy checkpoints have zero events, and collapsed plain has only one non-event. The Hanley-McNeil approximation recorded in `metrics.json` needs roughly 10 examples of each outcome for an AUROC of 0.70 to have a lower 95% bound above 0.5 at the pooled class counts; checkpoint-specific probability calibration would conventionally need about 100 events. At the observed collapsed-plain minority rate, roughly 180 eligible puzzles would be needed to expect 10 non-collapses. A healthy-checkpoint collapse rate near 1% would require roughly 1,000 eligible puzzles to expect 10 events.

More puzzles cannot repair the larger checkpoint-level limitation. There are four checkpoint families and only one independently collapsed family. A credible transfer claim needs several independent collapsed and stable training runs, with enough eligible puzzles in each target checkpoint to contain both outcomes. The puzzle-bootstrap intervals in `metrics.json` condition on these four checkpoints and must not be read as uncertainty over new checkpoints.

The label also asks only about loss of an already correct board. It does not cover puzzles still unsolved at iteration 128, cell-level deterioration, or collapse after iteration 1024.

## Verdict

**Not supported as a transferable early warning.** Early geometry cleanly identifies the known collapsed checkpoint in this sample, but checkpoint identity explains nearly the same held-out separation. Geometry does not survive the checkpoint-conditioned permutation test and fails when the collapsed checkpoint is held out. The evidence supports an early checkpoint difference, not a calibrated puzzle-level warning that generalizes to unseen checkpoints.

## Artifacts and reproduction

- `collection.json`: hashes, split assignments, checkpoint hashes, fixed early features, and separately defined outcomes
- `metrics.json`: all held-out metrics, bootstrap intervals, permutation nulls, calibration summaries, transfer results, and sample-size calculations
- `final_predictions.json`: final labels and calibrated probabilities
- `event_rates.png`, `model_comparison.png`, `calibration.png`, `geometry_coefficients.png`, `checkpoint_transfer.png`, and `index.html`: inspected visual artifacts
- `core.py`, `collect_early_warning.py`, `analyze_early_warning.py`, and `test_early_warning.py`: analysis code and seven focused tests

```sh
source venv/bin/activate
PYTHONWARNINGS='error::RuntimeWarning' python looping/trajectory_viz/study/10_early_warning/test_early_warning.py
PYTHONWARNINGS='error::RuntimeWarning' python looping/trajectory_viz/study/10_early_warning/analyze_early_warning.py --collection looping/trajectory_viz/study/10_early_warning/collection.json --output-dir looping/trajectory_viz/study/10_early_warning --permutations 500 --bootstraps 500
```
