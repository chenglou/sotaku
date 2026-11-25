# Sotaku

A plain **800K-parameter looped transformer** trained from scratch that solves **99.12%** of a 25,000-puzzle Sudoku-Extreme benchmark and **98.94%** of the full 422,786-puzzle test split. No halting head, no energy function, no row/column/box embeddings: just standard pre-norm blocks, 2D RoPE, one training trick for long-horizon stability, and FP32 inference.

**Paper:** [Sotaku: A Plain Looped Transformer Baseline Saturates Sudoku (PDF)](paper/sotaku_short.pdf) ([build & source notes](paper/README.md))

## Overview

- **Method:** later-iteration cross-entropy training + FP32 inference
- **Checkpoint:** [v2.0.0 weights, manifest, and validation records](https://github.com/chenglou/sotaku/releases/tag/v2.0.0)
- **Results (v2, FP32):** **99.12%** at 1024 iterations and **98.63%** at 4096 on the [25K benchmark](release/benchmark_25k.json); **98.94%** at 1024 on the full 422,786-puzzle `sapientinc/sudoku-extreme` test split
- **Model:** 4 shared-weight transformer blocks (width 128, 4 heads, MLP 512, 2D RoPE), 796,937 parameters
- **Training:** 2.7M puzzles, batch size 2048, peak LR `2e-3`, cosine decay, 50K steps, hard-to-easy curriculum

The model backpropagates through 16 iterations at a time, yet keeps improving past 1,000 iterations at inference. Under standard 16-iteration training, most runs eventually collapse if you unroll them that far. The fix is simple: on 20% of training batches, roll forward without gradients for `s ∈ {32, 64, 128, 256, 512}` iterations, detach the state, and apply the same 16-iteration cross-entropy loss from there. The other 80% of batches supervise iterations 1–16 as usual. That lets training reach iteration 528 while backpropagation never spans more than 16 steps.

See [looping/EXPERIMENTS_LOOPING.md](looping/EXPERIMENTS_LOOPING.md#additional-training-window-and-margin-penalty) for ablations with extra supervised windows and margin penalties.

### Training & Evaluation Notes

- **Curriculum & budget:** Reference 50K runs draw from the first 2.7M puzzles of the `sudoku-extreme` training split, widening the difficulty pool over time: rating 51+ to step 10K, 11+ to 20K, 1+ to 30K, then all ratings to 50K (102.4M puzzle presentations at BS=2048, with a 1,400-step warmup and cosine decay from `2e-3`).
- **Screening length:** Use 20K steps for routine comparisons and 50K to confirm; 10K runs do not preserve final rankings.
- **1,000-puzzle regime:** This repo focuses on the full training split; results in the 1,000-puzzle symmetry-augmented setting (46%–57% at 1024 iterations with a 10K mixed-difficulty schedule) are reported in Appendix J of [the paper](paper/sotaku_short.pdf).
- **Benchmark caveat:** The 25K benchmark samples 5,000 puzzles from each of five rating buckets (`0`, `1-2`, `3-10`, `11-50`, `51+`) out of the official test split. Because it guided model selection across the project, treat it as a dev set.

## Setup

Reference runs used Linux, Python 3.11, an NVIDIA H200, and PyTorch 2.10.0 + CUDA 12.8. Training requires a CUDA GPU with enough memory for batch size 2048 (gradient accumulation isn't implemented for later-iteration training) and runs in BF16 autocast. Inference defaults to FP32 with TF32 disabled because BF16 rounding degrades very long rollouts; CPU inference works too.

Install the CUDA 12.8 build (see [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/#v2100) for CPU-only):

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`requirements.txt` pins the core dependencies, `requirements-modal.txt` adds the CUDA 12.8 index, `requirements-dev.txt` adds tests/plotting, and `requirements-legacy.txt` preserves an earlier Linux snapshot.

## Train & Evaluate

To reproduce the 50K reference recipe (same preset and seed `20260730` as the released checkpoint):

```sh
python train.py --preset reference --seed 20260730 --run-name late_ce_50k

python -m iters.eval_more_iters runs/training/model_late_ce_50k.pt   --benchmark release/benchmark_25k.json --precision fp32 --iters 128 1024 2048 4096
```

For a faster 20K dev run, omit `--preset` (`development` is the default): `python train.py --run-name late_ce_20k`. Reusing a `--run-name` resumes from its saved checkpoint, so pick a fresh name for a new run.

Each `.pt` checkpoint ships with a matching `.pt.json` manifest (architecture config + checksum); keep the pair together. Evaluation outputs land in a new directory under `runs/`.

## Published Checkpoint

Download the v2 weights and manifest directly:

```sh
gh release download v2.0.0 --repo chenglou/sotaku --pattern 'model_late_state_ce.pt*'

python -m iters.eval_more_iters model_late_state_ce.pt   --benchmark release/benchmark_25k.json --precision fp32 --device cuda   --batch-size 256 --iters 128 1024 2048 4096
```

That solves `24,779 / 25,000` puzzles at 1024 iterations (99.12%) in ~16.5 minutes on an H200 (eager FP32, BS=256). Checksums and per-puzzle records are in [release/v2/README.md](release/v2/README.md).

To solve a single puzzle without downloading any dataset:

```sh
python solve.py model_late_state_ce.pt   '53..7.... 6..195... .98....6. 8...6...3 4..8.3..1 7...2...6 .6....28. ...419..5 ....8..79'
```

## Modal (Optional)

If you want to launch runs on Modal:

```sh
pip install modal==1.2.6
modal token new

# Reference 50K later-iteration training
modal run --detach looping/modal_health_methods.py --arm late-state-ce --seed 20260730 --name late_ce_50k

# Fetch weights from the Modal volume
modal volume ls sudoku-outputs
model=looping/model_late_ce_50k.pt
modal volume get sudoku-outputs "$model" .
modal volume get sudoku-outputs "$model.json" .

# Evaluate
modal run --detach modal_eval.py --model "$model" --precision fp32 --iters 128,1024,2048,4096
```

For 20K dev runs on Modal, use `modal run --detach looping/modal_stay_solved.py --arm late-state-ce --seed 20260730 --name late_ce_20k` (pass `--screen` for the older 10K screen). Launch each independent job in its own detached invocation.

## Visualizations

```sh
pip install matplotlib
python -m viz.visualize model_late_state_ce.pt --device cuda --n-iters 32
```

Outputs are written to `viz/output/`.

## Tests

Install `requirements-dev.txt`, then run both test suites (also run in CI):

```sh
python -m unittest discover -v
python scripts/run_research_tests.py
```

Research study tests run in isolated subprocesses because a few study folders share local module names.

## Key Files

- [paper/sotaku_short.pdf](paper/sotaku_short.pdf) & [paper/README.md](paper/README.md): paper draft and rebuild instructions.
- [stabilize/exp_testbed_20k.py](stabilize/exp_testbed_20k.py): shared model and training implementation.
- [looping/exp_health_methods.py](looping/exp_health_methods.py): 50K reference preset and matched recipe comparisons.
- [looping/exp_stay_solved.py](looping/exp_stay_solved.py): 20K dev preset and second-window experiments.
- [train.py](train.py), [solve.py](solve.py): CLI entrypoints for training and single-puzzle inference.
- [model_io.py](model_io.py), [inference.py](inference.py): checksum-verified model loading and shared recurrent inference loop.
- [iters/eval_more_iters.py](iters/eval_more_iters.py): iteration-scaling evaluation and per-puzzle logging.
- [checkpoint_utils.py](checkpoint_utils.py): checkpoint save/resume helpers.
- [looping/modal_health_methods.py](looping/modal_health_methods.py), [looping/modal_stay_solved.py](looping/modal_stay_solved.py), [modal_eval.py](modal_eval.py): optional Modal wrappers.

## Results

v2 checkpoint on the frozen 25K benchmark (eager FP32 on H200, batch size 256, evaluated 2026-09-02):

| Inference iterations | Solved puzzles | Accuracy |
|---:|---:|---:|
| 128 | 24,072 / 25,000 | 96.29% |
| 1024 | 24,779 / 25,000 | **99.12%** |
| 2048 | 24,762 / 25,000 | 99.05% |
| 4096 | 24,658 / 25,000 | 98.63% |

On the full 422,786-puzzle test split in FP32, v2 solves **98.94%** at 1024 iterations (see [the paper](paper/sotaku_short.pdf)). Precision and compilation choices matter a lot at high iteration counts; see [release/PRECISION_RESULTS.md](release/PRECISION_RESULTS.md).

The architecture is Sudoku-agnostic beyond knowing the board is a 2D grid (2D RoPE only, no row/column/box embeddings). Full scaling tables, stability diagnostics, and ablations are in [looping/EXPERIMENTS_LOOPING.md](looping/EXPERIMENTS_LOOPING.md) and [iters/EXPERIMENTS_ITERS.md](iters/EXPERIMENTS_ITERS.md).

## Research Notes

- [Training on later iterations](looping/EXPERIMENTS_LOOPING.md): repeated runs, second supervised windows, margin penalties, damping, and shorter training rollouts.
- [Iteration scaling](iters/EXPERIMENTS_ITERS.md): behavior at large inference iteration counts.
- [Recurrent normalization](stabilize/EXPERIMENTS_STABILIZE.md): RMSNorm, state-norm limits, and other stabilization attempts.
- [Evolution strategies](es/EXPERIMENTS_ES.md): ES fine-tuning of failed checkpoints and training from scratch.
- [Recurrent-state geometry](looping/trajectory_viz/study/README.md): PCA/projection study, adversarial controls, and why we stopped trusting trajectory plots.
- [Jacobian precision check](looping/spectral_diagnostics/RESULTS.md): FP64 autograd Jacobian check correcting the earlier low-precision finite-difference error.
- [Nearby-state and settling diagnostics](looping/basin_diagnostics/RESULTS.md): perturbation maps of healthy vs. failing width-128 checkpoints.
- [Weight-sharing study](looping/weight_tying/RESULTS.md): 18 paired runs comparing shared vs. independent weights, plus a fresh 10K-puzzle test set.
- [Training-window selection](looping/window_selection/RESULTS.md): 9 matched 20K runs comparing random offsets, confidence selection, and fixed offset 512.
- [Hyperloop study](looping/hyperloop/RESULTS.md) & [50K confirmation](looping/hyperloop_50k/RESULTS.md): four gated states raise mean 1024-iteration accuracy but don't eliminate collapse.
- [Attention residuals & gated message](looping/attention_residuals/RESULTS.md): Inkling 2 Mini loop design tested across 12 matched 20K runs; gated message never loses a solved puzzle but caps near 83–84%.
- [Baseline width](looping/width/README.md) & [width-160 50K confirmation](looping/width_50k/README.md): controlled comparisons of widths 160 and 192 against width 128.
- [Release verification](V2_RELEASE_AUDIT.md): v2 artifact provenance and audit.
- [Numerical sensitivity](release/PRECISION_RESULTS.md) & [burn-in dropout](looping/BURNIN_DROPOUT.md): inference precision checks and continuation ablations.

## Historical / Archived Code

Older Kaggle and pre-`sudoku-extreme` experiments kept for reference:

- `STALE_EXPERIMENTS_DOC.md` - chronological early experiment log (~200 runs)
- `arch/`, `recur/`, `curriculum/`, `misc/` - early architecture and curriculum experiments
- `pos_embedding/EXPERIMENTS_POS.md` - 2D RoPE and positional-encoding ablations
- `muon/EXPERIMENTS_MUON.md` - Muon optimizer experiments
- `rrn/RRN_EXPERIMENTS.md` - recurrent relational network baseline

Earlier helper scripts are preserved at the [v2.0.0 tag](https://github.com/chenglou/sotaku/tree/v2.0.0).
