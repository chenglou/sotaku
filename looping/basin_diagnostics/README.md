# Nearby-state and settling diagnostics

All six analyses are complete. See [results and interpretation](RESULTS.md) and the [full plot gallery](results/GALLERY.md).

The focused follow-up is now available: [high-resolution recursive zooms from the initial state](recursive/README.md). Those figures cover the first 1024 iterations and are separate from the 4096-iteration stability diagnostics below.

This study applies the initial-state perturbation idea from [Fractal basins trap latent reasoning](https://arxiv.org/abs/2609.04963) to existing Sotaku checkpoints. It does not train, fine-tune, select answers, or change inference defaults. The paper's [public code](https://github.com/GilpinLab/loopscape) was inspected at commit `fff8bf738663cd6ee7f62969b8e780e48c5a0db7`; no upstream code is executed or copied.

## Comparisons fixed before analysis

All three width-128 final checkpoints from the 20K study are included. Seeds 20260907 and 20260909 retain 97.028% and 92.960% accuracy at 4096; seed 20260908 falls from 94.312% at 1024 to 2.856% at 4096. All three 50K controls are included as a separate cohort: one has a moderate decline to 91.688%, and two remain above 97%. A moderate decline is not another example of catastrophic collapse. These labels come from the archived 25K evaluations, not from this diagnostic sample.

The model settings, final-update counts, weight checksums, and model-source checksums must match the archived records. Original weights and outputs are read-only. Every job records its source files, exact settings, environment, and array checksums. Completed analysis units resume only with matching identities; partial units can be recomputed. One detached invocation launches one worker.

## Measurements

The fixed protocol samples 60 benchmark puzzles, balanced across five rating buckets, excluding the first 200 per bucket used for training monitoring. Ten puzzles form a discovery set, and fifty form the quantitative analysis sample. These are not a new independent benchmark: the original 25K set has been reused during research. Perturbations are repeated measurements within a puzzle, not independent puzzles or training seeds.

For the fifty analysis puzzles, run ordinary inference to 4096 and compare small positive and negative perturbations in two orthogonal directions, applied at iterations 128 and 1024. Per-direction RMS is 0.1% or 3% of the reached hidden state's RMS. A zero-perturbation continuation accompanies every group. Compare each perturbation against its own zero continuation, since changing batch shapes can change floating-point execution. Report both rescues and harms, not a best-of-many selected answer.

Two discovery puzzles are used for the map gallery. The first is the first fixed-sample puzzle that remains correct in seed 20260907 but is lost between 1024 and 4096 in seed 20260908. The second is the first different puzzle that seed 20260907 solves after iteration 16. If either category is empty, use the next discovery puzzle and record the fallback. Gallery choices do not depend on how attractive a map looks. The same puzzles are used across all checkpoints.

For each gallery puzzle, use two independently seeded 2D planes, two perturbation scales, and both anchor iterations. Each 17x17 grid has an exact zero center and two extra unperturbed controls. Nearby states share the same frozen model. Feedback probabilities are recomputed from the perturbed hidden state; zero controls preserve the exact original probabilities. This matters because Sotaku's starting state contains the puzzle, unlike models with a separate constantly injected puzzle input. The study perturbs actual reached states instead of replacing the puzzle representation with arbitrary noise.

Each trajectory records first correct iteration, last answer change, correct-to-incorrect transitions, final correctness, and whether the final answer has remained unchanged for 128 further updates. Correctness follows the public evaluator: given digits are preserved in the displayed board, but the model's recurrent feedback is not clamped. An unchanged wrong answer is not a successful solve. A trajectory that returns to an earlier answer is timed by its last change, not by counting how often its final answer appeared.

Measurements start at the intervention. If a plot reports its starting iteration as the last answer change, no subsequent change occurred; the actual last change may have been earlier. A fixed percentage of state RMS is not a fixed absolute perturbation across iterations or checkpoints. State growth can make the same percentage much more disruptive later. The current comparison does not separate that scaling effect from changes in robustness to an equal absolute perturbation.

The sensitivity map measures the maximum normalized-hidden-state distance between neighboring pixels, sampled every 16 iterations, relative to their initial distance. This is not a Jacobian norm or a Lyapunov exponent. Finite measurements can miss short-lived separation. No numeric distance between Sudoku digit labels is used. PCA plots include three fixed neighboring trajectories and all stored times, without searching for a visually appealing projection; axes fitted in different maps are not directly comparable.

Settling-time differences at pixel separations 1, 2, and 4 are compared with shuffled spatial labels. Unconfirmed settling times are excluded only from that statistic and remain visible in all outcome maps. A 17x17 grid at two centered scales is an exploratory diagnostic, not sufficient evidence for a fractal dimension or mathematical chaos. All failed and nonfinite trajectories remain in outcome reporting.

## Running

```sh
source venv/bin/activate
python -m unittest looping.basin_diagnostics.test_diagnostics -v
modal run --detach looping/basin_diagnostics/modal_run.py --smoke
# Only after the GPU preflight and real-checkpoint smoke pass:
modal run --detach looping/basin_diagnostics/modal_run.py --key 20k_20260907
modal run --detach looping/basin_diagnostics/modal_run.py --key 20k_20260908
modal run --detach looping/basin_diagnostics/modal_run.py --key 20k_20260909
modal run --detach looping/basin_diagnostics/modal_run.py --key 50k_20260910
modal run --detach looping/basin_diagnostics/modal_run.py --key 50k_20260911
modal run --detach looping/basin_diagnostics/modal_run.py --key 50k_20260912
# Download committed figures and summaries; this does not launch any workers.
python -m looping.basin_diagnostics.collect
# Once all models are complete, verify artifacts and generate the gallery:
python -m looping.basin_diagnostics.report
```

Outputs live under `basin_diagnostics_v2_20260909/` on `sudoku-outputs`. `selection.json` fixes the puzzle identities. Each model saves `probe_summary.png`, sixteen map/PCA panels, raw arrays, logs, and a final manifest. The smoke has a shorter horizon and is not a stability result. Version 1 contains only the first passing smoke test; version 2 fixes the cross-model plotting helper and removes an unused batch-size setting before any full analyses. Actual batch sizes are 50 for unperturbed sample trajectories, 250 for paired probes, and 291 for a map including its controls. No release changes or training claims follow automatically from this study.
