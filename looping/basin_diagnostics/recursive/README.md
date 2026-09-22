# Recursive Settling-Time Maps

Precision correction, September 11: the original dense maps below used FP32 tensors but allowed reduced-precision matrix multiplication because a model import reset the precision setting. Their speckling is not reliable evidence of fractal geometry. The [deeper-zoom follow-up](deep/README.md#precision-correction) uses verified full FP32 and includes FP64 comparisons; an exact-size replay confirmed the difference.

Complete: [paired zooms with logarithmic colours](results_1024/recursive_comparison.png), [paired zooms with linear colours](results_1024/comparison.png), and the full settling-time/separation figures for the [healthy checkpoint](results_1024/20k_20260907/recursive.png) and [late-collapsing checkpoint](results_1024/20k_20260908/recursive.png).

The [deeper-zoom follow-up](deep/README.md) increases resolution to 401x401 and extends the saved 16x regions to 1024x magnification, with numerical precision checks. It retains the same 1024 model iterations.

This is the focused, higher-resolution follow-up to the initial coarse maps: one healthy and one late-collapsing width-128 checkpoint, one hard puzzle, and three freshly evaluated 201x201 grids per model at 1x, 4x, and 16x magnification. No training, model changes, PCA, benchmark sweep, or answer selection is included.

The completed maps show curved bands and fine-scale variation within the zoomed regions. All starts in these six maps reach the correct answer and remain unchanged over the final 128 iterations of the 1024-iteration window. This does not contradict the second checkpoint's known later collapse. These selected views illustrate structure; they do not establish a fractal dimension. The seven new grid/zoom tests and fifteen existing diagnostic tests passed in the GPU preflight. Row-overlap and zero-perturbation checks passed throughout both full runs, and downloaded artifacts passed their saved checksums.

The layout follows the recursive visualizations in [Fractal basins trap latent reasoning, Appendix E](https://arxiv.org/html/2609.04963v1#A5). Each figure has answer-settling time above the maximum separation between neighboring decoded outputs. Every panel uses its own explicitly labelled colour range. Boxes identify the next zoom; zooms are new evaluations, not enlarged images. The next box follows the largest mean neighboring settling-time difference among windows with at least 90% confirmed adjacent pairs, similar to the authors' [zoom routine](https://github.com/GilpinLab/loopscape/blob/fff8bf738663cd6ee7f62969b8e780e48c5a0db7/loopscape/fractal.py#L445).

Unlike EqR, Sotaku stores the puzzle in its initial hidden state rather than reinjecting the input each loop. The initial plane is therefore centered on the encoded puzzle, with random orthogonal perturbation directions. Replacing that state with random noise would discard the puzzle. Both models use the same puzzle, plane seed, and initial relative scale. The puzzle is the first difficulty-51+ benchmark row outside training-monitoring rows that both checkpoints solve at 512 but not 16, and that the healthy model retains at 4096 while the failing model loses it. Four small 33x33 previews through 512 iterations choose the common initial viewing range; these previews are framing aids, not long-run results.

The main maps evaluate every trajectory from initialization through iteration 1024 without early stopping. These visualize the initial solving process, not the later collapse already measured in the previous study. The checkpoint labels refer to the archived 4096-iteration evaluations; a model labelled late-collapsing can still solve every start within the shorter plotting window. A pixel records the last change to any of the 81 decoded digits. Gray means the answer changed within the final 128 iterations. Wrong answers can also become constant; each panel reports final exact-answer correctness separately. For agreement with the paper's decoded-output maps, these traces include all 81 raw output digits, without restoring given cells. The lower row uses the paper-style maximum Euclidean distance between neighboring decoded boards over time. This measurement depends on digit labels and is not a Jacobian norm or a formal Lyapunov exponent. No fractal dimension is estimated.

Eight-row chunks include one extra row on each side, so vertical neighbor comparisons are retained across chunk boundaries. Duplicate rows must produce identical final boards and last-change times. Two identical unperturbed controls accompany every chunk. Grid chunks are saved atomically with checksums and can resume after preemption. Model checksums, source files, settings, and selected puzzle are recorded. Existing artifacts remain untouched.

```sh
source venv/bin/activate
python -m unittest looping.basin_diagnostics.recursive.test_recursive -v
modal run --detach looping/basin_diagnostics/recursive/modal_run.py --key prepare
# After the framing and GPU preflight succeed, one detached launch per model:
modal run --detach looping/basin_diagnostics/recursive/modal_run.py --key 20k_20260907
modal run --detach looping/basin_diagnostics/recursive/modal_run.py --key 20k_20260908
python -m looping.basin_diagnostics.recursive.collect
# After both sequences finish, verify downloads and render the paired view:
python -m looping.basin_diagnostics.recursive.present
```

Outputs are under `basin_recursive_1024_v1_20260909/` on `sudoku-outputs`. Each model's `recursive.png` is updated after a completed zoom level. Local downloads live in `results_1024/`; raw arrays are excluded from Git and can be downloaded again with the collector. The two rows may follow different zoom locations after the shared initial view because each model's settling-time pattern can differ. The initial framing-only run is preserved at `basin_recursive_v1_20260909/` and locally in `results/`; no 4096-iteration high-resolution maps were launched there. App and call IDs are recorded in [jobs.json](jobs.json). Each dense sequence took about 17.5 minutes on an H200; both workers have exited.
