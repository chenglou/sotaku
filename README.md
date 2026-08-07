# Sotaku

From-scratch experiments on iterative neural Sudoku solvers. See [post](https://x.com/_chenglou/status/2032615500065419763)

## Current Status

- **Recommended default:** randomized detached late-state cross-entropy, with no architecture change or inference adjustment
- **Best demonstrated model:** late-state cross-entropy through step 39K, then a second future cross-entropy window plus a minimum-margin loss through step 50K
- **Benchmark:** `sapientinc/sudoku-extreme` via `load_dataset(..., split="test")`
- **Best demonstrated result:** **99.00%** at 1024 and **98.46%** at 2048, without inference-time damping
- **Simple late-state result:** **98.80%** at 1024 and **92.98%** at 2048, without inference-time damping
- **Published checkpoint:** the historical plain-backprop checkpoint remains available at **98.9%** at 1024
- **Simplest reliable alternative:** recurrent RMSNorm reached **91.3-92.4%** at 1024 across three seeds
- **Architecture:** 4-layer shared-weight transformer, 2D RoPE, ~800K params
- **Training setup:** 2.7M-puzzle pool, BS=2048, LR=2e-3, 16 supervised iterations, 50K optimizer steps, cosine decay, reverse curriculum

Sotaku's 796,937-parameter looped transformer is trained through 16 differentiable iterations at a time but can continue improving for more than 1,000 iterations at inference. The recommended recipe leaves the model and loss unchanged. On 20% of batches, it first runs without gradients to iteration 32, 64, 128, 256, or 512, detaches that state, and applies the usual 16-iteration averaged cross-entropy from there. The other 80% of batches use the ordinary iterations 1-16.

The best combined checkpoint first used that recipe through step 39K. For the final 11K steps, sampled late-state batches added a second supervised window after a detached gap and a calibrated loss that protects the weakest correct-answer margin. This staged result is the current accuracy leader, but it has only one seed and should be treated as experimental. ES remains useful for selected collapsed checkpoints, but polishing an already-good model did not help.

### Training And Data Budget

Current SOTA experiments use the first 2.7M puzzles from the `sudoku-extreme` training split. Batches sample from progressively broader rating pools: rating 21+ through step 10K, 6+ through 20K, 1+ through 30K, then all 2.7M puzzles through 50K. At batch size 2048, a 50K run processes 102.4M sampled puzzle presentations. Full evaluation uses a separate balanced set of 25,000 test puzzles, with 5,000 from each rating bucket.

Use the 20K schedule by default when comparing new ideas. It produced three healthy late-state-CE runs and is much faster than 50K, although its final 16-iteration accuracy is about 3-5 points lower. Promote promising results to 50K before treating them as reliable: positive or negative phase changes may still appear after 20K. The 10K screens did not preserve the ranking observed at 20K and 50K, so they are not reliable model-selection proxies. We have not run a clean reduced-puzzle-count ablation for the current SOTA recipe; shorter runs still draw from the same 2.7M-puzzle pool.

Reducing recurrent-state exposure is a different experiment. A margin-only run that never used a training state beyond iteration 128 produced a checkpoint scoring 98.18% at 1024, but all runs capped at 80, 128, or 192 eventually collapsed before step 50K. Short late-state exposure can produce a strong checkpoint, but broad exposure through iteration 800 was needed in that experiment for reliable final weights.

Detailed evidence lives in `iters/EXPERIMENTS_ITERS.md`, `stabilize/EXPERIMENTS_STABILIZE.md`, `looping/EXPERIMENTS_LOOPING.md`, and `es/EXPERIMENTS_ES.md`.

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the recommended 50K late-state-CE recipe
python -c "from looping.exp_stay_solved import train; train(arm='late_state_ce', full_50k=True, run_name='loop_late_state_ce_50k')"

# Evaluate the final checkpoint without inference adjustments
python -c "from iters.eval_more_iters import evaluate; evaluate('model_loop_late_state_ce_50k.pt', exp_module='stabilize.exp_testbed_20k', iter_counts=[128, 1024, 2048, 4096])"
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

# Recommended 50K late-state-CE training
modal run --detach looping/modal_stay_solved.py --arm late-state-ce --full-50k

# Experimental combined SOTA: first produce the exact trial-0 source checkpoint.
modal run --detach looping/modal_stay_solved.py --arm control --full-50k --trial 0

# After that job commits its step-39K checkpoint, continue with recheck CE + margin.
modal run --detach looping/modal_late_switch.py --mode margin-floor5

# Inspect outputs on the volume
modal volume ls sudoku-outputs
modal volume get sudoku-outputs looping/model_loop_stay_late_state_ce_50k_trial0.pt .

# Evaluate the saved model
modal run --detach modal_eval.py --exp stabilize.exp_testbed_20k --model looping/model_loop_stay_late_state_ce_50k_trial0.pt --iters 128,1024,2048,4096
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

- `looping/exp_stay_solved.py` - recommended 50K late-state-CE recipe and second-window variants
- `looping/modal_stay_solved.py` - detached Modal launcher for the recommended recipe
- `looping/modal_late_switch.py` - exact step-39K branch used by the combined SOTA
- `looping/exp_health_methods.py` - matched vanilla, RMSNorm, late-state CE, consistency, and margin ablations
- `looping/exp_late_supervision.py` - 20K late-state screening and horizon ablations
- `looping/eval_trajectory_geometry.py` - recurrent-update dimension, smoothness, and cross-puzzle basis diagnostics
- `iters/exp_baseline_lr2e3.py` - historical 98.9% released-checkpoint recipe; difficult to reproduce reliably
- `looping/eval_late_recipe.py` - optional delayed-damping evaluation for models that still deteriorate deeply
- `stabilize/exp_lr2e3_outer_rmsnorm.py` - simpler reliable alternative with no extra burn-in forwards
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
- `looping/eval_delayed_damping.py` + `looping/EXPERIMENTS_LOOPING.md` - inference-time damping rescue, loop schedules, residual scaling, and late-state training
- `es/exp_es_finetune.py` + `es/EXPERIMENTS_ES.md` - optional evolution-strategies rescue or polish for unconstrained checkpoints
- `stabilize/EXPERIMENTS_STABILIZE.md` - training-time stabilization study (recurrent RMSNorm and caps, burn-in, weight averaging, feedback noise)

## Results

All percentages below use the balanced 25K-puzzle test set.

| Model | 128 iterations | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| **staged late-state CE + recheck CE + margin, final** | 96.26% | **99.00%** | **98.46%** | **82.17%** |
| **late-state CE only, final** | **96.47%** | 98.80% | 92.98% | 43.70% |
| historical released plain-backprop checkpoint | 95.3% | 98.9% | 98.8% | - |
| late stay-consistency switch, best deep checkpoint | 95.91% | 97.14% | 90.50% | 35.22% |
| recurrent RMSNorm, three selected checkpoints | 90.81-91.58% | 91.32-92.44% | 91.37-92.55% | - |
| randomized late states + delayed damping, three finals | 93.61-94.36% | 94.74-96.84% | 94.71-96.88% | 94.69-96.91% |

The model is sudoku-agnostic in the sense that it only assumes a 2D grid: no row, column, or box constraint embedding, just 2D RoPE in attention. Full scaling tables, stability analysis, interventions, and ablations live in [looping/EXPERIMENTS_LOOPING.md](looping/EXPERIMENTS_LOOPING.md) and [iters/EXPERIMENTS_ITERS.md](iters/EXPERIMENTS_ITERS.md).

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
