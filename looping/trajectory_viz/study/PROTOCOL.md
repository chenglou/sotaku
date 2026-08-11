# Sotaku recurrent-geometry study protocol

This study tests predefined, interpretable hypotheses about Sotaku's recurrent hidden states. It is not a search for visually appealing projections.

## Models

Use the same four checkpoints as `eval_trajectory_geometry.py` whenever the hypothesis permits:

- naturally stable plain backpropagation
- collapsed plain backpropagation
- standalone late-state cross-entropy
- staged late-state cross-entropy plus recheck and margin protection

## Data split

Sample balanced puzzles with a fixed seed and divide them before fitting anything:

- discovery: fit projections or choose candidate statistics
- validation: choose among variants defined before examining the final split
- final holdout: evaluate the selected analysis once

Use at least 20 puzzles per split when Modal cost permits, with equal representation from the five rating buckets. Never fit a projection on cells or iterations later reported as held out. Record any smaller sample explicitly.

## Required controls

Every reported structure must include the relevant controls:

- shuffled labels for supervised axes
- shuffled iteration order for temporal structure
- random orthonormal projections or a matched-rank random subspace for projection-sensitive claims
- transfer to unseen puzzles
- transfer across at least two checkpoints, unless the claim is specifically a checkpoint difference
- comparison against an unconstrained categorical probe when claiming an ordered or cyclic representation

Report held-out effect sizes rather than relying on visual appearance. A plot may illustrate a result but cannot establish one.

## Outputs

Each analysis owns one numbered directory under this folder. It must contain:

- analysis code and focused tests
- machine-readable metrics
- inspectable PNG or HTML artifacts
- a concise `REPORT.md` stating the hypothesis, controls, held-out result, limitations, and verdict

Use plain language. Distinguish exploratory observations from results that survived the protocol.
