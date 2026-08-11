# ARM 01: predictive-uncertainty coordinate

## Verdict

**Inconclusive, with partial evidence for decodable uncertainty.** A discovery-fit linear axis predicts final-split entropy beyond exact iteration identity in all four checkpoints. The final partial R-squared values are `.385`, `.098`, `.167`, and `.225` for stable plain, collapsed plain, late-state CE, and combined margin. Removing the eight-dimensional output-head contrast row space changes those values only slightly.

The stronger hypothesis is not established. The durable run uses cell-centered hidden states without removing per-cell magnitude, so it cannot rule out accumulated state norm. It also lacks the protocol-required comparison with an unconstrained categorical probe. Late-state CE does not beat the shuffled-label control, and the fitted axes show only weak temporal order. The result supports “predictive uncertainty is linearly accessible from hidden state” more than “the model has a norm-independent ordered uncertainty coordinate.”

## Hypothesis

Recurrent hidden states contain a one-dimensional ordered coordinate for predictive uncertainty or candidate entropy, rather than only iteration identity or accumulated state norm.

This run operationalizes predictive uncertainty with three label-free functions of the nine output probabilities:

- normalized softmax entropy, treated as the primary measure;
- the top-one/top-two probability gap;
- one minus the maximum probability.

Rule-derived Sudoku candidate entropy was not evaluated.

## Data and fitting

The sample contains 60 canonical test puzzles, 12 from each rating bucket `0`, `1–2`, `3–10`, `11–50`, and `51+`. A fixed sample seed (`20260811`) and within-bucket split seed (`20260812`) assign four puzzles per bucket to each of discovery, validation, and final. The audit confirmed that the three 20-puzzle splits are disjoint and contain all 60 unique puzzle hashes.

The analysis uses originally blank cells at recurrent iterations `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024`. Each puzzle has equal total weight. At each puzzle and snapshot, the mean hidden state over all 81 cells is subtracted. A ridge axis is fit on discovery puzzles after subtracting discovery iteration means from both features and the uncertainty target. Ridge strength is chosen on validation separately for each checkpoint and target. The fixed axis is then evaluated once on final puzzles.

All four prescribed checkpoints were evaluated:

- naturally stable plain backpropagation;
- collapsed plain backpropagation;
- standalone late-state cross-entropy;
- staged late-state cross-entropy plus recheck and margin protection.

## Final held-out result

The table reports the primary normalized-entropy result. “Output-null” removes the centered output-head contrast row space before fitting. Label and random columns show the 95th percentile of the corresponding 64-control distribution. Axis order is the equal-puzzle-weighted fraction of ordered snapshot pairs that move in the entropy-decreasing direction.

| Checkpoint | Final partial R² beyond iteration | Output-null R² | Shuffled-label p95 | Random-axis p95 | Label-shuffle p | Axis order | Final blank-cell accuracy at 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stable plain | .3846 | .3828 | .3186 | .00009 | .0154 | .5797 | 1.000 |
| Collapsed plain | .0983 | .0953 | .0600 | .01737 | .0154 | .5058 | .828 |
| Late-state CE | .1669 | .1587 | .1925 | .00237 | 1.0000 | .5961 | 1.000 |
| Combined margin | .2247 | .2233 | .2029 | .00068 | .0154 | .5805 | .960 |

The two sensitivity targets give the same broad ranking. Final top-two-gap R-squared is `.369`, `.097`, `.161`, and `.215`; one-minus-maximum-probability R-squared is `.358`, `.088`, `.160`, and `.211` in the same checkpoint order.

Stable plain, collapsed plain, and combined margin exceed every sampled entropy label-shuffle axis. Late-state CE falls below every label-shuffle axis: its observed `.1669` is below the shuffled distribution's `.1872–.1925` 5th–95th percentile range. All four fitted entropy axes exceed every sampled random orthonormal axis. With only 64 controls, the smallest plus-one value is `1/65 = .0154`. A Holm correction across the four primary checkpoint comparisons makes the three smallest label-shuffle values `.0615`; the run therefore does not supply family-wise `p < .05` evidence.

Projecting out the direct digit-readout contrast span reduces entropy R-squared by only `.0014–.0082`. The signal is not confined to the output head's linear contrast directions, although a nonlinear or redundant readout remains possible.

## Temporal control

The actual predictive-entropy paths are mostly ordered: their all-pairs decreasing fractions are `.840`, `.832`, `.796`, and `.740`. The fitted hidden axes are much less ordered at `.580`, `.506`, `.596`, and `.581`.

Against shuffled iteration orders, the raw plus-one values for fitted-axis entropy order are `.092`, `.400`, `.031`, and `.046`. None survives correction across checkpoints. The collapsed checkpoint is essentially at chance. Temporal order is therefore weak evidence and does not independently establish an ordered coordinate.

The snapshot-level plot shows where the pooled effect comes from. Stable plain rises to `.856` at iteration 32. Collapsed plain rises to `.523` at iteration 512 and falls to `.119` at 1024. Late-state CE reaches `.914` at iteration 128. At nearly deterministic late snapshots, the uncertainty variance can be zero or extremely small: partial R-squared is then undefined or becomes a very large negative number. The revised figure shows a `−.30` display floor with downward markers and exact outlier ranges instead of compressing the useful portion of the plot.

## Controls and protocol audit

Completed controls:

- whole-puzzle discovery, validation, and final splits, balanced by rating;
- unseen-puzzle transfer with discovery-only projection fitting;
- transfer across all four checkpoints;
- exact iteration-identity baseline;
- 64 discovery-label shuffles within each iteration;
- 64 random orthonormal one-dimensional axes;
- 64 shuffled iteration orders for temporal structure;
- output-head contrast removal as a leakage sensitivity;
- equal total weight per puzzle.

Missing controls:

- **State norm:** cell centering removes board-wide translation but does not normalize each cell vector or include norm as a nuisance variable. The run therefore does not test the “not merely accumulated state norm” clause.
- **Unconstrained categorical probe:** the protocol requires this comparison before calling a representation ordered. No entropy-bin categorical model was fit.

Because these two missing controls are central to the stated hypothesis, the durable run does not meet the confirmatory verdict rule recorded in `DESIGN.md`. All three uncertainty targets were also evaluated on final rather than selecting one target globally on validation; this report treats entropy as primary and the other two as sensitivity checks instead of choosing the strongest final result.

## Limitations

- There are 20 puzzles per split, the protocol minimum, and one checkpoint per training regime rather than independent training seeds.
- Cell snapshots are not independent observations. Equal puzzle weighting prevents puzzles with more blanks from dominating, but the run does not provide puzzle-level intervals.
- Predictive entropy is computed from the model's own logits, so linear accessibility from hidden state is expected to some degree. Output-null projection makes the result less trivial but does not make it causal.
- The shuffled-label null is not centered near zero for every checkpoint. This exposes checkpoint-specific distribution shift between discovery and final; late-state CE in particular fails this control.
- The analysis is observational. It does not intervene on the fitted axis to change uncertainty or solving behavior.
- No rule-derived candidate set or candidate entropy was measured.

## Artifacts and verification

- `entropy_v1_20260811/metrics.json`: machine-readable configuration, puzzle hashes and splits, validation candidates, final metrics, complete null samples, per-iteration results, and accuracies.
- `entropy_v1_20260811/heldout_encoding.png`: held-out effects, output-head removal, shuffled labels, and random axes.
- `entropy_v1_20260811/trajectory_monotonicity.png`: fitted-axis order, actual uncertainty order, and shuffled-time ranges.
- `entropy_v1_20260811/iteration_profiles.png`: held-out snapshot profiles with unstable late values explicitly clipped and annotated for display.
- `entropy_v1_20260811/run.log`: durable run progress and completion record.
- `run_entropy_study.py`, `entropy_core.py`, and `modal_entropy.py`: analysis and detached Modal entry point.
- `test_entropy.py`: focused statistical, split, control, and durable-artifact tests.

The three PNGs were opened at full resolution. The held-out and monotonicity figures have readable labels and visible null markers. The original iteration plot failed visual inspection because late negative outliers compressed all other curves; it was rerendered from the unchanged JSON with a documented display floor. The rerendered figure was opened again and checked for clipping, overlaps, legibility, and faithful outlier annotation.

Local verification:

```bash
source venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=looping/trajectory_viz/study/01_entropy \
  python -m unittest discover \
  -s looping/trajectory_viz/study/01_entropy -p 'test_*.py' -v
```

All 13 tests pass. The Modal wrapper uses one `.spawn()` call, retries preempted work, commits the output volume in `finally`, and is intended to be launched with `modal run --detach`.
