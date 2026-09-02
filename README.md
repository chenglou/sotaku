# Sotaku

From-scratch experiments on iterative neural Sudoku solvers. See [post](https://x.com/_chenglou/status/2032615500065419763)

## Current Status

- **Recommended default:** randomized detached late-state cross-entropy, with no architecture change or inference adjustment
- **Benchmark:** `sapientinc/sudoku-extreme` via `load_dataset(..., split="test")`
- **Recommended checkpoint result:** **98.80%** at 1024 and **92.98%** at 2048, without inference-time damping
- **Published checkpoint:** the historical plain-backprop checkpoint remains available at **98.9%** at 1024; the recommended late-state checkpoint is not yet published
- **Architecture:** 4-layer shared-weight transformer, 2D RoPE, ~800K params
- **Training setup:** 2.7M-puzzle pool, BS=2048, LR=2e-3, 16 supervised iterations, 50K optimizer steps, cosine decay, reverse curriculum

Sotaku's 796,937-parameter looped transformer is trained through 16 differentiable iterations at a time but can continue improving for more than 1,000 iterations at inference. The recommended recipe leaves the model and loss unchanged. On 20% of batches, it first runs without gradients to iteration 32, 64, 128, 256, or 512, detaches that state, and applies the usual 16-iteration averaged cross-entropy from there. The other 80% of batches use the ordinary iterations 1-16. Training therefore reaches iteration 528, with gradients spanning only the final 16 iterations of each sampled trajectory.

Late-state CE is the main recipe for its simplicity and reliability across tested seeds. Additional rechecks and margin losses remain [research experiments](looping/EXPERIMENTS_LOOPING.md#staged-recheck-and-margin), not part of the recommended recipe. Their results and reproduction commands are preserved in the experiment notes.

### Training And Data Budget

The 50K reference recipe uses the first 2.7M puzzles from the `sudoku-extreme` training split. Batches sample from progressively broader rating pools: rating 51+ through step 10K, 11+ through 20K, 1+ through 30K, then all ratings through 50K. At batch size 2048, a 50K run processes 102.4M sampled puzzle presentations. The learning rate starts with a 1,400-step warmup, then follows cosine decay from 2e-3.

Use 20K steps for routine comparisons and 50K to confirm promising results; important changes may still appear after 20K. The 10K screens did not preserve the later ranking. Both schedules draw from the same 2.7M-puzzle pool; a smaller training dataset has not been tested with the current recipe.

Reported full evaluations use 25,000 test puzzles: 5,000 from each rating bucket, 0, 1-2, 3-10, 11-50, and 51+. Monitoring and final reporting reuse the official test split, with some overlap between their samples. These benchmark results informed model selection; they are not an untouched final holdout.

## Setup

Reference training used Linux, Python 3.11, an NVIDIA H200, and PyTorch 2.10.0 with CUDA 12.8. Training requires a CUDA GPU and substantial memory at batch size 2048; gradient accumulation is not yet supported for late-state training. CPU inference is supported, but the reported scores use CUDA bfloat16 evaluation.

For the reference CUDA build, following the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2100):

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-modal.txt
```

For CPU-only inference, install the CPU build from the linked instructions instead. `requirements-modal.txt` contains the smaller core dependency set; `requirements.txt` is an older full environment snapshot. Neither file fully locks the reference environment.

## Train And Evaluate

The 98.80% checkpoint used the 50K late-state-CE preset with seed `20260730`. The following command uses that preset and seed; independent retraining is not guaranteed to match its exact score.

```sh
python -c "from looping.exp_health_methods import train; train(arm='late_state_ce', random_seed=20260730, run_name='loop_late_state_ce_50k')"

python -m iters.eval_more_iters model_loop_late_state_ce_50k.pt \
  --exp stabilize.exp_testbed_20k --iters 128 1024 2048 4096
```

For a 20K development run instead:

```sh
python -c "from looping.exp_stay_solved import train; train(arm='late_state_ce', random_seed=20260730, run_name='loop_late_state_ce_20k')"
```

## Published Checkpoint

If you want the published 98.9% result without retraining:

```sh
gh release download baseline-lr2e3-checkpoint --pattern model_baseline_lr2e3.pt

python -m iters.eval_more_iters model_baseline_lr2e3.pt \
  --exp iters.exp_baseline_lr2e3 --iters 1024
```

Expected result: `24728/25000` solved, or `98.9%`, at `1024` test-time iterations.

## Modal (Optional)

The core training code is provider-agnostic. This launcher uses the reference 50K preset and seed `20260730`:

```sh
pip install modal
modal token new

# Reference 50K late-state-CE training
modal run --detach looping/modal_health_methods.py --arm late-state-ce --trial 0

# Inspect outputs on the volume
modal volume ls sudoku-outputs
model=looping/model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt
modal volume get sudoku-outputs "$model" .

# Evaluate the saved model
modal run --detach modal_eval.py --exp stabilize.exp_testbed_20k --model "$model" --iters 128,1024,2048,4096
```

For 20K development on Modal instead, use `modal run --detach looping/modal_stay_solved.py --arm late-state-ce --no-screen`. That launcher uses seed `20260724` for trial 0. Keep `--no-screen` explicit: omitting it selects the 10K screen. Modal installs dependencies from `requirements-modal.txt`, whose PyTorch version remains unpinned; the local installation above does not pin the remote image.

## Visualizations

Using the published checkpoint:

```sh
pip install matplotlib
python -m viz.visualize model_baseline_lr2e3.pt --exp iters.exp_baseline_lr2e3 --device cuda --n-iters 32

python viz/plot_iteration_scaling.py
```

Outputs go to `viz/output/`.

## Key Files

- [stabilize/exp_testbed_20k.py](stabilize/exp_testbed_20k.py): shared model and training implementation.
- [looping/exp_health_methods.py](looping/exp_health_methods.py): 50K reference preset and matched training-method comparisons.
- [looping/exp_stay_solved.py](looping/exp_stay_solved.py): 20K development preset and optional second-window experiments.
- [iters/eval_more_iters.py](iters/eval_more_iters.py): iteration-scaling evaluation for the plain architecture used by the recommended and published checkpoints.
- [checkpoint_utils.py](checkpoint_utils.py): checkpoint saving and resumption.
- [looping/modal_health_methods.py](looping/modal_health_methods.py), [looping/modal_stay_solved.py](looping/modal_stay_solved.py), and [modal_eval.py](modal_eval.py): optional Modal wrappers.

## Results

The recommended recipe and reference results below use the balanced 25K-puzzle test set. Recheck and margin results are kept in the [research notes](looping/EXPERIMENTS_LOOPING.md#staged-recheck-and-margin).

| Model | 128 iterations | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| **late-state CE only, final (recommended)** | 96.47% | **98.80%** | 92.98% | 43.70% |
| historical released plain-backprop checkpoint | 95.3% | 98.9% | 98.8% | - |
| recurrent RMSNorm, three selected checkpoints | 90.81-91.58% | 91.32-92.44% | 91.37-92.55% | - |
| randomized late states + delayed damping, three finals | 93.61-94.36% | 94.74-96.84% | 94.71-96.88% | 94.69-96.91% |

The model is sudoku-agnostic in the sense that it only assumes a 2D grid: no row, column, or box constraint embedding, just 2D RoPE in attention. Full scaling tables, stability analysis, interventions, and ablations live in [looping/EXPERIMENTS_LOOPING.md](looping/EXPERIMENTS_LOOPING.md) and [iters/EXPERIMENTS_ITERS.md](iters/EXPERIMENTS_ITERS.md).

## Research Notes

- [Looping and late-state training](looping/EXPERIMENTS_LOOPING.md): reliability cohorts, rechecks, margin losses, damping, and reduced training horizons.
- [Iteration scaling](iters/EXPERIMENTS_ITERS.md): the original checkpoint and long-horizon behavior.
- [Recurrent normalization](stabilize/EXPERIMENTS_STABILIZE.md): RMSNorm, caps, and other stabilization experiments.
- [Evolution strategies](es/EXPERIMENTS_ES.md): checkpoint rescue, polish, and training from scratch.
- [Recurrent-state geometry](looping/trajectory_viz/study/README.md): visualization study, controls, and limitations.

## Historical / Archived Code

Older Kaggle and pre-`sudoku-extreme` experiments are preserved for reference, but they are not the current public path:

- `STALE_EXPERIMENTS_DOC.md` - archived chronological experiment log
- `arch/`, `recur/`, `curriculum/`, `misc/` - older experiment families
- `pos_embedding/EXPERIMENTS_POS.md` - 2D RoPE introduction and positional-encoding ablations
- `muon/EXPERIMENTS_MUON.md` - Muon optimizer experiments
- `rrn/RRN_EXPERIMENTS.md` - RRN experiments
- root-level scripts such as `eval_extreme.py`, `eval_only.py`, and `eval_difficulties.py` - archival only
