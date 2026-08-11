# Sudoku-constraint geometry in Sotaku's recurrent states

## Verdict

Sotaku's hidden states make current Sudoku constraint quantities linearly readable on unseen puzzles, especially at iteration 16. This supports a narrow claim that the state contains row, column, box, peer-conflict, and candidate information. The stronger hypothesis does not survive the full protocol: in the three healthy checkpoints, iteration and certainty variables already explain almost all current-conflict variation; hidden-state geometry adds little prediction of later constraint improvement; the future effect does not beat shuffled iteration; and fitted probes mostly fail direct cross-checkpoint transfer.

The failing collapsed checkpoint is the exception. Its states carry a large constraint residual and some future-improvement signal, but the signal is not stable under the temporal control. The most constraint-specific geometry in this run is therefore associated with collapse rather than shared successful solving.

## Predefined targets

All targets are computed before probe fitting from the model's predictions and the puzzle givens. Given cells are clamped to their supplied digits.

- Expected row, column, box, and unique-peer conflicts sum the probability that a cell and a relevant peer choose the same digit.
- Hard conflict counts use the argmax-predicted board. Row, column, and box counts each range from 0 to 8; the unique-peer count ranges from 0 to 20.
- Dynamic candidate counts are the digits absent from the current hard-predicted row, column, box, or union of 20 unique peers.
- Static candidate count uses givens only.
- Correct-digit legality asks whether the known solution digit is absent from the current hard-predicted peers. Prediction correctness is reported separately.
- The primary future target is the reduction in expected unique-peer conflicts between one sampled state and the next sampled state. Positive values mean improvement.

Numeric count probes test the predefined order directly. The corresponding categorical probes use unrestricted one-hot class targets, so they do not assume that adjacent counts should be adjacent in representation space.

## Protocol

The saved collection contains 60 test puzzles sampled with seed `20260811`, equally divided among the five rating buckets. Each bucket contributes four puzzles to discovery, four to validation, and four to final holdout, giving 20 whole puzzles per split. States and logits were saved at iterations `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024` for the four canonical checkpoints.

Ridge probes are fit on discovery puzzles, select one of `0.01, 1, 100, 10000` on validation, and are evaluated on final puzzles. The direct target probes use iterations 16, 128, and 1024. Controls use 16 shuffled-label fits and 16 Haar-random subspaces with rank matched to the learned probe. Aggregate temporal trends use 2,000 shuffled iteration orders.

The stricter residual probes pool sampled times while giving each puzzle-time pair equal total weight. The current-conflict nuisance model contains exact sampled iteration, maximum probability, top-two probability margin, and entropy. The future-improvement nuisance model also contains current expected and hard conflicts, candidate deviation, and correctness. The state coefficient is fit only to the remaining discovery residual. Its learned scalar axis is compared with shuffled labels, a matched rank-one random projection, and hidden states whose sampled iterations are shuffled within every puzzle-cell trajectory.

Cross-checkpoint transfer applies the complete source-checkpoint standardization, nuisance model, and state probe to another checkpoint's final puzzles without refitting.

## Held-out results

At iteration 16, expected unique-conflict probes reach final-holdout `R² = .860, .935, .865, .866` for stable plain, collapsed plain, late-state CE, and combined margin. Row, column, and box probes are similarly strong (`R² = .859–.945`). Every observed value exceeds the maximum shuffled-label and matched-rank random value in its corresponding 16-repeat control. The [numeric probe plot](numeric_probe_transfer.png) shows that this is an early and middle-state result; late healthy targets become nearly constant after the puzzles solve.

The ordered hard-conflict probes are weaker but positive at iteration 16: unique-conflict count gives `R² = .268, .458, .403, .418`. Unconstrained hard-conflict classification gives balanced accuracy `.294, .300, .275, .264`, versus approximately `.20` for shuffled labels. Dynamic candidate classification gives `.449, .467, .431, .457`, versus approximately `.333` for shuffled labels. The [categorical plot](categorical_probe_transfer.png) omits late points with only one supported class; those cases cannot establish geometry.

This readability is mostly not independent of time and certainty in healthy checkpoints:

| checkpoint | nuisance `R²`, current conflict | state partial `R²` | nuisance `R²`, future improvement | state partial `R²`, future |
|---|---:|---:|---:|---:|
| stable plain | .980 | .0041 | .373 | -.0012 |
| collapsed plain | .471 | .248 | .299 | .097 |
| late-state CE | .987 | .0095 | .377 | .011 |
| combined margin | .983 | .0045 | .348 | .0061 |

The healthy future gains are negligible. The collapsed checkpoint's `.097` future partial `R²` is larger than its shuffled-label maximum `.011` and random rank-one maximum `.040`, but shuffled iteration reaches `.107`; its one-sided 16-repeat value is `p = .118`. Shuffled iteration also exceeds the observed future gain in every healthy checkpoint. The [residual-probe plot](future_improvement_prediction.png) shows these controls together.

Direct transfer provides no shared-axis replication. Off-diagonal current-conflict partial `R²` is positive only for late-state CE to collapsed plain (`.039`); off-diagonal future transfer is positive only for stable plain to collapsed plain (`.019`). Most transfers are negative, several catastrophically so. This result is visible in the [cross-checkpoint matrix](cross_checkpoint_transfer.png). The test is deliberately strict: separately trained representations may be rotated even when they encode the same abstract variable.

The behavioral trajectories themselves are coherent. Stable plain, late-state CE, and combined margin solve all 20 final puzzles by iteration 128 and remain solved at 1024. Collapsed plain solves 18/20 at 128 but only 2/20 at 1024; expected unique conflicts rise to `.846`. Row, column, and box components move together rather than showing a single privileged unit type. See [constraint dynamics](constraint_dynamics.png) and [component dynamics](constraint_components.png).

## Interpretation

The early hidden state is rich enough to decode continuous constraint burden and moderately decode hard conflict or candidate classes. That signal is real on unseen puzzles and is not a generic low-rank projection artifact. However, the protocol does not show that a shared constraint-satisfaction axis organizes successful recurrent computation. In healthy models the direct signal is almost redundant with iteration and certainty, it contributes little to forecasting the next constraint reduction, and it does not transfer unchanged across checkpoints.

The results are consistent with constraint information being distributed through a broader prediction/certainty representation. They are also consistent with each checkpoint using a different linear coordinate system. The analysis does not establish a constraint-specific mechanism or causal role.

## Limits

- There is one checkpoint per training regime and only 20 final puzzles, although every split is difficulty-balanced.
- Sixteen label and projection controls give a minimum plus-one value of `1/17 ≈ .0588`; they are effect-size controls, not high-resolution significance tests.
- Cell observations within a puzzle are dependent. Splitting and weighting operate at the puzzle level, but the reported linear fits are not population-level inference over independent cells.
- Only 12 log-spaced iterations are observed. “Next snapshot” spans different iteration gaps, handled by exact iteration indicators but not by dense trajectory sampling.
- Certainty and current-status nuisance variables are downstream model outputs. Removing them is a deliberately strict conditional test and may remove genuine constraint computation along with confounding signal.
- Direct cross-checkpoint transfer assumes aligned feature coordinates. A discovery-only alignment could test equivalence up to rotation, but was not fitted here.
- Linear readout and observational change do not establish causality. Activation intervention would be needed to test whether a decoded constraint axis controls solving.
- Near-zero late target variance makes some numeric `R²` values undefined or extremely negative. The JSON preserves those results; plots clip values below `-.5` and omit one-class categorical points.

## Artifacts and reproduction

- Machine-readable results: [`metrics.json`](metrics.json), [`probe_metrics.csv`](probe_metrics.csv), [`residual_probes.csv`](residual_probes.csv), [`cross_checkpoint_transfer.csv`](cross_checkpoint_transfer.csv), [`dynamics.csv`](dynamics.csv), [`paired_changes.csv`](paired_changes.csv), and [`trend_controls.csv`](trend_controls.csv)
- Inspectable gallery: [`index.html`](index.html)
- Informative calibration at iteration 16: [`probe_calibration_16.png`](probe_calibration_16.png)
- Collector and durable log: [`collect_constraints.py`](collect_constraints.py), [`modal_constraints.py`](modal_constraints.py), and [`constraints_states_v1.log`](constraints_states_v1.log)
- Analysis and tests: [`analyze_constraints.py`](analyze_constraints.py) and [`test_constraints.py`](test_constraints.py)

The collector entrypoint uses one `.spawn()` call and is intended to be invoked with `modal run --detach`. This completion pass did not launch Modal; it downloaded the already completed `constraints_states_v1.pt` payload and reran analysis locally.

Reproduce the local pass from the repository root:

```bash
source venv/bin/activate
MPLCONFIGDIR=/tmp/sotaku_constraints_mpl python -m unittest looping/trajectory_viz/study/03_constraints/test_constraints.py
MPLCONFIGDIR=/tmp/sotaku_constraints_mpl python looping/trajectory_viz/study/03_constraints/analyze_constraints.py looping/trajectory_viz/study/03_constraints/constraints_states_v1.pt --output-dir looping/trajectory_viz/study/03_constraints
```

All final PNGs were opened at rendered resolution. Labels, legends, control markers, clipping notes, and heatmap annotations were checked after correcting degenerate late-target scales.
