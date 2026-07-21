# Sotaku

From-scratch experiments on iterative neural Sudoku solvers. See [post](https://x.com/_chenglou/status/2032615500065419763)

## Current Status

- **Highest-scoring model:** `iters/exp_baseline_lr2e3.py`
- **Recommended training recipe:** `stabilize/exp_lr2e3_outer_rmsnorm.py`
- **Benchmark:** `sapientinc/sudoku-extreme` via `load_dataset(..., split="test")`
- **Best demonstrated result:** **98.9%** puzzle accuracy at 1024 test-time iterations
- **Recommended-recipe result:** **91.3-92.4%** at 1024 across three independently trained models; all three were healthy
- **Architecture:** 4-layer shared-weight transformer, 2D RoPE, ~800K params
- **Training setup:** BS=2048, LR=2e-3, 16 training iterations, cosine decay, reverse curriculum

Sotaku reaches 98.9% puzzle accuracy at 1024 test-time iterations with an ~800K-parameter looped transformer trained for only 16 iterations, showing that its learned computation can keep improving far beyond the training horizon. The recommended `stabilize/exp_lr2e3_outer_rmsnorm.py` recipe makes strong long-horizon behavior substantially more reproducible: all three full-schedule runs finished healthy, and their harvested checkpoints scored 91.3-92.4% at 1024 and 91.6-92.8% at 2048 over 25,000 test puzzles. The unconstrained recipe remains the path to the highest demonstrated scores, while RMSNorm is the reliable default; direct ES reached 94-95%, and the settling fine-tune in `es/exp_es_settle.py` reached 96.8% at 1024 and stayed essentially flat through 4096. See `iters/EXPERIMENTS_ITERS.md`, `stabilize/EXPERIMENTS_STABILIZE.md`, and `es/EXPERIMENTS_ES.md` for the full evidence.

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the current SOTA model
python iters/exp_baseline_lr2e3.py

# Evaluate the trained checkpoint at 1024 test-time iterations
python -c "from iters.eval_more_iters import evaluate; evaluate('model_baseline_lr2e3.pt', exp_module='iters.exp_baseline_lr2e3', iter_counts=[1024])"
```

## Reproduce The Released Checkpoint

If you want the published 98.9% result without retraining:

```sh
gh release download baseline-lr2e3-checkpoint --pattern model_baseline_lr2e3.pt

python -c "from iters.eval_more_iters import evaluate; evaluate('model_baseline_lr2e3.pt', exp_module='iters.exp_baseline_lr2e3', iter_counts=[1024])"
```

Expected result: `24728/25000` solved, or `98.9%`, at `1024` test-time iterations.

## Modal (Optional)

The core training code is provider-agnostic. For Modal:

```sh
pip install modal
modal token new

# Train on Modal and keep the job alive if your client disconnects
modal run --detach modal_run.py --exp iters.exp_baseline_lr2e3

# Inspect outputs on the volume
modal volume ls sudoku-outputs
modal volume get sudoku-outputs model_baseline_lr2e3.pt .

# Evaluate the saved model
modal run modal_eval.py --exp iters.exp_baseline_lr2e3 --model model_baseline_lr2e3.pt --iters 1024
```

Experiments must expose `train(output_dir=".")`. Modal-specific deps are in `requirements-modal.txt`.

## Visualizations

Main figure entry points:

```sh
# Attention maps, confidence evolution, entropy, head specialization
python viz/visualize.py model_baseline_lr2e3.pt --exp iters.exp_baseline_lr2e3 --device cuda --n-iters 32

# Iteration-scaling summary plots from the recorded experiment tables
python viz/plot_iteration_scaling.py

# Collapse diagnostics comparing multiple checkpoints
python viz/plot_collapse_diagnostics.py \
  model_baseline_lr2e3.pt model_baseline_lr3e3.pt model_baseline_lr1e3.pt \
  --exps iters.exp_baseline_lr2e3 iters.exp_baseline_lr3e3 iters.exp_baseline_lr1e3 \
  --output-dir viz/output
```

Outputs go to `viz/output/`.

For GPU-backed collapse diagnostics on Modal:

```sh
modal run --detach viz/modal_viz.py
modal volume get sudoku-outputs viz_diagnostics/ viz/output/
```

## Blessed Entry Points

- `iters/exp_baseline_lr2e3.py` - highest-scoring but unreliable training recipe
- `stabilize/exp_lr2e3_outer_rmsnorm.py` - recommended reliable training recipe; keep its best long-horizon checkpoint
- `stabilize/eval_lr2e3_outer_rmsnorm.py` - full 16/128/1024/2048 evaluation of harvested RMSNorm checkpoints
- `stabilize/exp_lr2e3_outer_cap1.py` - less invasive cap-1 alternative
- `stabilize/eval_lr2e3_outer_cap1.py` - full evaluation of harvested cap-1 checkpoints
- `iters/eval_more_iters.py` - canonical evaluation across test-time iteration counts
- `analyze_failures_new.py` - per-iteration failure analysis for current models
- `checkpoint_utils.py` - checkpoint discovery and config-checked resume
- `modal_run.py` - minimal Modal training wrapper
- `modal_eval.py` - Modal wrapper for `iters/eval_more_iters.py`
- `modal_analyze.py` - Modal wrapper for analysis utilities
- `viz/visualize.py` - attention/confidence/head-specialization figures for current models
- `viz/plot_collapse_diagnostics.py` - hidden-state and prediction-stability diagnostics
- `viz/plot_iteration_scaling.py` - static summary plots from the documented scaling tables
- `iters/EXPERIMENTS_ITERS.md` - current source of truth for iteration-scaling results
- `es/exp_es_finetune.py` + `es/EXPERIMENTS_ES.md` - optional evolution-strategies rescue or polish for unconstrained checkpoints
- `stabilize/EXPERIMENTS_STABILIZE.md` - training-time stabilization study (recurrent RMSNorm and caps, burn-in, weight averaging, feedback noise)

## Results

| Model | Params | Training Time | Accuracy |
|-------|--------|---------------|----------|
| **exp_baseline_lr2e3 (1024 test iters)** | 800K | ~2h40m (H200) | **98.9%** |
| exp_baseline_lr2e3 (16 test iters) | 800K | ~2h40m (H200) | 81.8% |
| exp_lr2e3_outer_rmsnorm (harvested, 1024 test iters) | 800K | ~2h45m (H200) | 91.3-92.4% |
| exp_lr2e3_outer_cap1 (harvested, 1024 test iters) | 800K | ~3h (H200) | 91.3-91.9% |
| [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (reference) | 7M | ~18h (L40S) | ~87% |

The model is sudoku-agnostic in the sense that it only assumes a 2D grid: no row/col/box constraint embedding, just 2D RoPE in attention. Running more test-time iterations than used during training is the key result: 16 training iterations scales cleanly to 1024 evaluation iterations. Full scaling tables, stability analysis, interventions, and ablations live in [iters/EXPERIMENTS_ITERS.md](iters/EXPERIMENTS_ITERS.md).

## Auxiliary Utilities

- `test_data.py` - comparison helper for loading `test.csv` directly; not the canonical benchmark path
- `logs_to_tensorboard.py` - historical log conversion helper
- `tensorboard_utils.py` - lightweight TensorBoard logger used by a few older experiments
- `viz/` - plotting and visualization scripts for model behavior

## Historical / Archived Code

Older Kaggle and pre-`sudoku-extreme` experiments are preserved for reference, but they are not the current public path:

- `STALE_EXPERIMENTS_DOC.md` - archived chronological experiment log
- `arch/`, `recur/`, `curriculum/`, `misc/` - older experiment families
- `pos_embedding/EXPERIMENTS_POS.md` - 2D RoPE introduction and positional-encoding ablations
- `muon/EXPERIMENTS_MUON.md` - Muon optimizer experiments
- `rrn/RRN_EXPERIMENTS.md` - RRN experiments
- root-level scripts such as `eval_extreme.py`, `eval_only.py`, and `eval_difficulties.py` - archival only
