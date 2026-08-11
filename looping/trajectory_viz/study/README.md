# Recurrent-state geometry study

This study asked twelve independent questions about Sotaku's recurrent hidden state. Every analysis used the same four checkpoints, balanced puzzle splits, frozen analysis choices, unseen-puzzle evaluation, shuffled controls, and checkpoint-transfer tests where the question allowed them. See [`PROTOCOL.md`](PROTOCOL.md) for the shared rules.

## Bottom line

The model's state contains real, compact Sudoku information, but it does not follow one universal geometric object. We found no shared helix, loop, arc, oscillation, raw fixed point, solvedness direction, or transferable collapse-warning direction.

The recurring picture is simpler:

- Recurrent motion is smooth in both healthy and collapsed models.
- The state progressively represents the current answer, candidate-set size, cell position, conflicts, answer margin, and solve progress.
- Digit identity occupies a compact categorical subspace, but the digits do not have a privileged numeric or cyclic order. This is expected because Sudoku is unchanged by a global relabeling of the nine digits.
- Healthy models keep changing in raw state space, but their normalized state direction settles and their answers remain stable.
- The collapsed checkpoint also moves smoothly. Its normalized direction keeps drifting until some correct-answer margins cross the output boundary.
- Most useful coordinates are checkpoint-local. A direction fitted in one checkpoint generally does not remain the same direction in another.

This supports interpreting Sotaku as a smooth, puzzle-dependent computation with shared local features, rather than as traversal of one rigid low-dimensional shape.

## Results

| Analysis | Result | Representative artifact |
|---|---|---|
| [01. Uncertainty](01_entropy/REPORT.md) | Some uncertainty is linearly readable beyond iteration count, but the evidence is not strong enough for an ordered uncertainty axis. | [held-out encoding](01_entropy/entropy_v1_20260811/heldout_encoding.png) |
| [02. Answer margin](02_margin/REPORT.md) | Margin is decoded very accurately within each checkpoint and its history predicts later failure. The exact axis does not transfer between checkpoints and random low-rank subspaces perform similarly. | [boundary risk](02_margin/margin_boundary_v1_20260811/primary_boundary_risk.png) |
| [03. Sudoku constraints](03_constraints/REPORT.md) | Current conflicts and candidate information are strongly readable. Healthy checkpoints add little future-improvement signal after accounting for time and confidence, and direct axes do not transfer. | [constraint dynamics](03_constraints/constraint_dynamics.png) |
| [04. Solve progress](04_progress/REPORT.md) | A robust board-level progress coordinate transfers to unseen puzzles. It mostly measures generic solve time rather than an exact count of wrong cells. | [progress trajectories](04_progress/progress_trajectories.png) |
| [05. Digit symmetry](05_digit_symmetry/REPORT.md) | Digit categories form a compact rank-8 subspace. Natural `1` through `9` order and cyclic order do not beat shuffled orders. | [order controls](05_digit_symmetry/order_controls.png) |
| [06. Cell roles](06_cell_roles/REPORT.md) | Candidate-set size and row, column, and box roles emerge during recurrence after controlling for the input symbol. | [candidate geometry](06_cell_roles/artifacts/cell_roles_v1/candidate_size_geometry.png) |
| [07. Difficulty](07_difficulty/REPORT.md) | The state orders model solve latency in healthy checkpoints, but does not robustly encode the dataset's human-oriented rating. | [held-out predictions](07_difficulty/artifacts/protocol_v2/heldout_predictions.png) |
| [08. Temporal modes](08_temporal_modes/REPORT.md) | The dynamics are highly predictable and smooth, but persistence is usually better than a fitted linear dynamical model. There is no useful oscillatory or rotational mode. | [temporal statistics](08_temporal_modes/curvature_fourier_summary.png) |
| [09. Settling](09_fixed_point/REPORT.md) | Healthy models do not approach a raw fixed point. They approach a stable direction in normalized state space; the collapsed model keeps turning. | [settling and accuracy](09_fixed_point/settling_and_accuracy.png) |
| [10. Early warning](10_early_warning/REPORT.md) | Pooled geometry identifies the collapsed checkpoint family, not which unseen puzzle will later collapse. No transferable puzzle-level warning was established. | [confounded comparison](10_early_warning/model_comparison.png) |
| [11. Causal axes](11_causal_axes/REPORT.md) | Moving along an oracle true-digit output direction causally changes answer margin, but does not reliably improve long-term solving. A learned solvedness direction fails the causal controls. | [dose response](11_causal_axes/dose_response.png) |
| [12. Adversarial audit](12_adversarial_audit/REPORT.md) | Projection selection can manufacture convincing arcs, loops, and helices from synthetic null data. The shared-shape claims fail the final held-out audit. | [adversarial gallery](12_adversarial_audit/adversarial_gallery.png) |

## Practical use

For future health diagnostics, prefer quantities that remained meaningful under controls: weakest correct-answer margin, normalized-state movement, relative acceleration, solved-answer retention, and output stability. Do not use an attractive PCA path, raw update size, smoothness alone, or proximity to a raw fixed point as evidence that a checkpoint is healthy.

The study does not identify a new training loss by itself. It does support the existing root-cause direction: train models to preserve correct decisions and to compute from genuinely late states, then use damping or ES only as optional polish rather than as the explanation for why the model works.

## Scope

The study used 60 held-out test puzzles per arm: 20 for discovery, 20 for validation, and 20 touched once for the final result, with four puzzles from each rating bucket in every split. The checkpoint set contains one plain stable model, one plain collapsed model, one late-state-CE model, and one combined margin model. Conclusions about collapse transfer therefore need independent collapsed training runs before they can be treated as general laws.

Large activation tensors and projection-search caches are intentionally kept out of Git. Each arm includes the code, frozen choices, tests, reports, summary metrics, and plots needed to inspect the reported conclusion.
