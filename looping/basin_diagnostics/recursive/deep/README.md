# Deeper Recursive Zooms

Complete: [paired deeper zooms](results/deep_comparison.png), full settling-time/separation figures for the [healthy](results/20k_20260907/deep_zoom.png) and [late-collapsing](results/20k_20260908/deep_zoom.png) checkpoints, and their [healthy](results/20k_20260907/precision_comparison.png) / [late-collapsing](results/20k_20260908/precision_comparison.png) precision comparisons.

Both detached workers completed on September 11, 2026, taking 117.3 and 116.4 minutes on separate H200s. All eight 401x401 grids and eight precision checks are downloaded and checksum-verified. All 28 GPU preflight tests, row-overlap checks, and duplicate controls passed. No workers remain active. App and call IDs are in [jobs.json](jobs.json).

The [slow-puzzle follow-up](../../slow_puzzles/README.md) screens a larger puzzle sample and renders selected slow and easy cases fully in FP64, including the initial encoding.

## Results

The higher-resolution views reveal curved bands and narrower bands inside the selected regions. At 1024x magnification, the selected regions mainly show a few broad bands with fine texture, not an unambiguous repeating fractal. Every dense-grid start and every FP64 check reached the correct answer and stayed unchanged through iteration 1024; all dense-grid last changes occurred by iteration 410. These are early-solving pictures, not new 4096-iteration stability evaluations.

| Zoom | Healthy last-change range | Healthy FP32/FP64 exact-time agreement | Late-collapsing last-change range | Late-collapsing exact-time agreement |
|---|---|---|---|---|
| 16x | 33-382 | 88.7% | 36-116 | 94.8% |
| 64x | 36-358 | 69.4% | 36-129 | 90.9% |
| 256x | 52-358 | 55.8% | 52-116 | 75.3% |
| 1024x | 64-410 | 30.8% | 54-116 | 60.8% |

Agreement is measured on 441 shared starting coordinates per zoom. The broad patterns remain recognizable in FP64, but exact times become increasingly precision-sensitive. At 1024x, median differences are 4 and 0 iterations; 90th-percentile differences are 56 and 5. The FP32 audit batches reproduce the dense-grid times exactly at all sampled coordinates. Thus the finest texture cannot be confidently attributed to fractal geometry. No fractal dimension was estimated.

## Precision Correction

The original dense maps allowed reduced-precision matrix multiplication despite using FP32 tensors. The renderer selected `highest` before loading the model, but importing the training module silently reset the setting to `high`. The saved original environment confirms `high`. The new workers import the model before selecting `highest`, and their saved environments confirm full FP32 matrix multiplication. This also changes the initial encoder's arithmetic; the checkpoint, puzzle, and plane definition are unchanged.

An exact-size replay of original grid rows 96-103, including both halo rows, reproduced all original settling times with the old precision settings. Switching to full FP32 reproduced all new-map times at the same coordinates. Only 13.5% of those times agree across the two precision settings, although every final board agrees. The new FP32 maps show much cleaner bands, and their 16x FP64 audits agree on 88.7% and 94.8% of exact settling times for the healthy and late-collapsing models. See [precision_history.json](precision_history.json). The earlier speckling should not be interpreted as fractal detail.

## Method

Follow-up to the [initial recursive maps](../README.md). The same two width-128 checkpoints, puzzle 294840, initial hidden-state plane, and 1024-iteration window are retained. No training, inference adjustments, or new checkpoint selection is involved.

Each model starts at its saved 16x region, rerendered at 401x401 resolution instead of 201x201. Subsequent fresh grids zoom to 64x, 256x, and 1024x. Each next box follows the original rule: largest mean neighboring settling-time difference among boxes with sufficient settled points. The plots show last answer-change time and peak distance between neighboring decoded boards. Every trajectory runs the full 1024 iterations; gray marks changes within the final 128 iterations. Correctness is recorded separately.

At every zoom, a 21x21 subset is rerun in FP32 and FP64. Both retain the same full-FP32 encoded puzzle, plane directions, and RMS scale. FP64 constructs the perturbations and performs the recurrence in double precision. The comparison reports settling-time agreement, absolute differences, correlation, and final-answer agreement. A separate comparison checks the effect of the smaller FP32 audit batch. Rounding at initialization is measured relative to one dense-grid pixel. These checks distinguish reproducible broad structure from precision-sensitive fine detail; disagreement alone does not establish the absence of chaotic dynamics.

The earlier source files and artifacts are checksum-verified and remain untouched. New source hashes, checkpoint identities, chunk checksums, repeated row halos, and duplicate unperturbed controls are checked. Each model runs in its own detached Modal invocation with resumable chunks. A completed result is accepted only after its artifact checksums pass.

The completed study used the sources in commit `83050da`. A later portability fix allows `rtol=atol=1e-12` for duplicate FP64 hidden states on CPU; decoded answers and GPU controls still require exact agreement. Saved measurements are unchanged, and source checks prevent resuming old runs with modified code.

```sh
source venv/bin/activate
python -m unittest looping.basin_diagnostics.recursive.deep.test_deep looping.basin_diagnostics.recursive.test_recursive -v
modal run --detach looping/basin_diagnostics/recursive/deep/modal_run.py --key 20k_20260907
modal run --detach looping/basin_diagnostics/recursive/deep/modal_run.py --key 20k_20260908
python -m looping.basin_diagnostics.recursive.deep.collect
# After both workers complete:
python -m looping.basin_diagnostics.recursive.deep.present
```

Outputs: `basin_deep_zoom_v1_20260911/` on `sudoku-outputs`. Figures are qualitative views of one puzzle, not a fractal-dimension measurement or a comparison of long-horizon checkpoint reliability.
