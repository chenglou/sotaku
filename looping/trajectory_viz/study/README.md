# Recurrent-state geometry study

This study asked twelve independent questions about Sotaku's recurrent hidden state. Every analysis used the same four checkpoints, balanced puzzle splits, frozen analysis choices, unseen-puzzle evaluation, shuffled controls, and checkpoint-transfer tests where the question allowed them. See [`PROTOCOL.md`](PROTOCOL.md) for the shared rules.

## Bottom line

The model's state contains real, compact Sudoku information, but it does not follow one universal geometric object. We found no shared helix, loop, arc, oscillation, hidden-state fixed point, direction that reliably improves solving when used to modify the state, or transferable warning of later accuracy loss.

The recurring picture is simpler:

- Recurrent motion is smooth in both healthy and collapsed models.
- The state progressively represents the current answer, candidate-set size, cell position, conflicts, answer margin, and solve progress.
- Digit identity occupies a compact categorical subspace, but the digits do not have a privileged numeric or cyclic order. This is expected because Sudoku is unchanged by a global relabeling of the nine digits.
- The accurate models' complete states keep moving, but their directions change slowly at the measured later iterations and their answers remain correct.
- The failing checkpoint also moves smoothly. Its state direction keeps changing until an incorrect digit scores above the correct digit in some cells.
- Most useful coordinates are specific to one checkpoint. A direction fitted in one model generally does not work unchanged in another.

This supports a smooth computation that depends on the puzzle, with some features shared across puzzles, rather than traversal of one rigid low-dimensional shape.

## Reading The Reports

- A **probe** here is a small regression or classifier fitted to the hidden states to predict a quantity, such as current cell accuracy. It does not change the Sudoku model. This differs from the 1K-puzzle monitoring evaluations called probes in older training logs.
- An **axis** is a direction in hidden-state space. A **subspace** is a collection of such directions. Being able to predict a quantity from those directions does not establish that changing them will improve solving.
- The **correct-answer margin** is the correct digit's score minus the highest incorrect digit's score. A negative margin means an incorrect digit outranks the correct one. Some separate controls use the top-two predicted scores without consulting the answer key; the reports distinguish them.
- A **residual** is what remains after subtracting a stated effect. For example, an iteration-controlled regression asks whether hidden states predict anything beyond knowing the iteration number.
- **Discovery**, **validation**, and **final** refer to separate puzzle sets used to fit the analysis, choose its settings, and evaluate it once. They are not additional Sudoku training stages.
- Plot labels such as `stable_plain` and `collapsed_plain` identify the selected checkpoints. They do not guarantee that the corresponding training recipes always work or fail.

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
| [09. State changes](09_fixed_point/REPORT.md) | The accurate models' hidden states keep moving, but their directions change slowly at the measured later iterations. The failing model keeps turning faster. | [state changes and accuracy](09_fixed_point/settling_and_accuracy.png) |
| [10. Early warning](10_early_warning/REPORT.md) | Pooled geometry identifies the collapsed checkpoint family, not which unseen puzzle will later collapse. No transferable puzzle-level warning was established. | [confounded comparison](10_early_warning/model_comparison.png) |
| [11. Causal interventions](11_causal_axes/REPORT.md) | A state change chosen using the answer key raises the correct digit's score relative to alternatives, but does not reliably improve later solving. A learned direction that predicts current accuracy also fails to reliably improve it when used to change the state. | [dose response](11_causal_axes/dose_response.png) |
| [12. Adversarial audit](12_adversarial_audit/REPORT.md) | Projection selection can manufacture convincing arcs, loops, and helices from synthetic null data. The shared-shape claims fail the final held-out audit. | [adversarial gallery](12_adversarial_audit/adversarial_gallery.png) |

## Practical use

For future diagnostics, useful measurements are the weakest correct-answer margin, normalized-state movement, relative acceleration, the fraction of solved puzzles that remain solved, and prediction changes. An attractive PCA path, small raw updates, smoothness alone, or proximity to a hidden-state fixed point did not reliably distinguish accurate checkpoints in this study.

These observations motivate testing training that preserves correct predictions or directly supervises later iterations.

## Scope

The study used 60 test puzzles per analysis: 20 for discovery, 20 for validation, and 20 touched once for the final result, with four puzzles from each rating bucket in every split. The checkpoint set contains one accurate original model, one failing original model, one model trained on later iterations, and one combined model. [The protocol](PROTOCOL.md#models) maps these descriptions to the historical plot labels and defines the combined recipe. Conclusions about failure across models need independently trained failing checkpoints before they can be treated as general laws.

Large activation tensors and projection-search caches are intentionally kept out of Git. Each analysis includes code, frozen choices, tests, reports, summary metrics, and plots for inspecting its conclusion.
