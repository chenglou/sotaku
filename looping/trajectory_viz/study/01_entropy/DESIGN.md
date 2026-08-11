# ARM 01 locked analysis design

This design was fixed before evaluating the final split.

> Execution audit: the durable `entropy_v1_20260811` run used the same sample,
> split, snapshots, checkpoints, targets, and three-way fitting discipline, but
> it did not execute this full design. It used cell-centered states without
> unit normalization, selected ridge strength separately for every target and
> checkpoint, ran 64 controls, and did not add the state-norm nuisance model,
> categorical comparison, bootstrap, global variant selection, or Holm
> correction specified below. `REPORT.md` therefore treats the durable result
> as incomplete evidence and does not apply the confirmatory verdict rule.

## Hypothesis

Recurrent cell states contain a one-dimensional ordered coordinate for the model's predictive uncertainty. The coordinate must explain held-out, within-iteration variation after controlling for state norm; it cannot be established by iteration identity, accumulated magnitude, or a fitted visualization alone.

## Sample and split

- Load 12 puzzles from each of the five canonical rating buckets with seed `20260811`.
- Shuffle within rating bucket, then assign four puzzles per bucket to discovery, validation, and final splits. Each split therefore contains 20 complete puzzles.
- Use only cells that were blank in the original puzzle.
- Use snapshots `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024`.
- Give every puzzle equal total weight.
- Run the four checkpoints from `looping.eval_trajectory_geometry.DEFAULT_MODELS`.

## Candidate analyses

The candidate targets are normalized softmax entropy, one minus maximum probability, and the top-one/top-two probability gap. The candidate responses are cell-centered unit hidden-state direction and the same response after removing the centered output-head contrast row space. These choices remove hidden-state magnitude by construction.

For each checkpoint and candidate, fit a discovery-only ridge axis after removing exact iteration means. Choose ridge strength from the fixed grid `1e-4, 1e-3, 1e-2, 1e-1, 1, 10` on validation. Choose one target/response pair globally by the median validation gain beyond an exact-iteration plus within-iteration log-state-norm model across the four checkpoints. Do not refit the selected axes after validation.

## Final metrics and controls

Evaluate the selected analysis once on final puzzles. Report:

- partial R-squared beyond exact iteration;
- partial R-squared from state norm beyond exact iteration;
- incremental partial R-squared from the fitted axis beyond exact iteration and state norm;
- within-iteration weighted correlation;
- a discovery-fit forward encoding comparison between one ordered target degree of freedom and unrestricted five-bin categorical target coding;
- 199 within-puzzle, within-iteration shuffled-label axes;
- 128 random orthonormal one-dimensional axes;
- 999 shuffled iteration-order controls for temporal monotonicity;
- fixed-prediction, rating-stratified puzzle bootstrap intervals.

The categorical bins are discovery quintiles. The same bin boundaries are applied to validation and final observations. Exact iteration indicators and within-iteration standardized log centered-state norm are nuisance variables in ordered and categorical encoding models.

## Verdict rule

A checkpoint counts as support when the final axis has positive incremental partial R-squared beyond iteration and norm, beats both shuffled-label and random-axis nulls after Holm correction across the four checkpoints, and its ordered encoding captures at least half of the held-out categorical encoding gain. The study supports the cross-checkpoint hypothesis only if at least two checkpoints count as support. Temporal monotonicity is reported as a control but is not required, because a faithful uncertainty coordinate may reverse when a checkpoint collapses.
