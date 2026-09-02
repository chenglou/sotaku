# Sotaku

From-scratch experiments on iterative neural Sudoku solvers. See [post](https://x.com/_chenglou/status/2032615500065419763)

## Current Status

- **Recommended default:** randomized detached late-state cross-entropy, with no architecture change or inference adjustment
- **Benchmark:** [frozen 25K-puzzle sample](release/benchmark_25k.json) from the `sapientinc/sudoku-extreme` test split
- **Recommended checkpoint result:** **99.12%** at 1024 and **98.63%** at 4096 with FP32 inference, without inference-time damping
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

Reference training used Linux, Python 3.11, an NVIDIA H200, and PyTorch 2.10.0 with CUDA 12.8. Training requires a CUDA GPU and substantial memory at batch size 2048; gradient accumulation is not yet supported for late-state training. Training uses BF16 autocast. Inference defaults to FP32 with TF32 matmul disabled because reduced precision can substantially degrade very long trajectories. CPU inference is supported; reported benchmark scores use CUDA.

For the reference CUDA build, following the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2100):

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only inference, install the CPU build from the linked instructions instead. `requirements.txt` pins the core packages; `requirements-modal.txt` also selects the CUDA 12.8 build. Optional tests and plotting dependencies are in `requirements-dev.txt`. The old Linux environment snapshot is preserved as `requirements-legacy.txt`. Evaluations record the complete installed environment, including transitive dependencies and the GPU driver.

## Train And Evaluate

The recommended checkpoint used the 50K late-state-CE preset with seed `20260730`. Its unchanged weights score 99.12% at 1024 in FP32, versus the historical 98.80% in BF16. The following command uses that preset and seed; independent retraining is not guaranteed to match its exact score.

```sh
python train.py --preset reference --seed 20260730 --run-name late_ce_50k

python -m iters.eval_more_iters runs/training/model_late_ce_50k.pt \
  --benchmark release/benchmark_25k.json --precision fp32 --iters 128 1024 2048 4096
```

For a 20K development run, use `python train.py --run-name late_ce_20k`; `development` is the default preset. Reusing a run name resumes its saved optimizer and schedule. Use a new name for an independent run.

New inference weights have an adjacent `.pt.json` manifest with the exact model settings and checksum. Keep both files together. The evaluator uses the same recurrence as training, including normalization or layer schedules, and saves per-puzzle results in a new directory under `runs/`. Unlabelled historical weights require an explicit `--legacy-defaults --exp stabilize.exp_testbed_20k` for a known plain model; the published v1 file is recognized by its checksum.

## Published Checkpoint

If you want the published 98.9% result without retraining:

```sh
gh release download baseline-lr2e3-checkpoint --pattern model_baseline_lr2e3.pt

python -m iters.eval_more_iters model_baseline_lr2e3.pt \
  --benchmark release/benchmark_25k.json --precision bf16 --iters 1024
```

The fresh pinned-environment evaluation solves `24719/25000` at 1024 iterations (98.876%). The original published count was `24728/25000` (98.912%); both round to 98.9%. Use CUDA BF16, eager execution, and batch size 256 to reproduce the fresh evaluation.

For one puzzle, without downloading the dataset:

```sh
python solve.py model_baseline_lr2e3.pt \
  '53..7.... 6..195... .98....6. 8...6...3 4..8.3..1 7...2...6 .6....28. ...419..5 ....8..79'
```

The output is the model's proposed completion, with the given digits preserved.

## Modal (Optional)

The core training code is provider-agnostic. This launcher uses the reference 50K preset and seed `20260730`:

```sh
pip install modal==1.2.6
modal token new

# Reference 50K late-state-CE training
modal run --detach looping/modal_health_methods.py --arm late-state-ce --seed 20260730 --name late_ce_50k

# Inspect outputs on the volume
modal volume ls sudoku-outputs
model=looping/model_late_ce_50k.pt
modal volume get sudoku-outputs "$model" .
modal volume get sudoku-outputs "$model.json" .

# Evaluate the saved model
modal run --detach modal_eval.py --model "$model" --precision fp32 --iters 128,1024,2048,4096
```

For 20K development on Modal, use `modal run --detach looping/modal_stay_solved.py --arm late-state-ce --seed 20260730 --name late_ce_20k`. The launcher now defaults to 20K; the older 10K screen requires `--screen`. Launch each independent job in its own detached invocation. Evaluation outputs use unique directories under `evaluations/` on the volume.

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
- [train.py](train.py), [solve.py](solve.py): short entrypoints for the recommended training recipe and single-puzzle inference.
- [model_io.py](model_io.py), [inference.py](inference.py): checksum-verified inference artifacts and the shared recurrent inference path.
- [iters/eval_more_iters.py](iters/eval_more_iters.py): iteration-scaling evaluation, frozen benchmark indices, and per-puzzle records.
- [checkpoint_utils.py](checkpoint_utils.py): checkpoint saving and resumption.
- [looping/modal_health_methods.py](looping/modal_health_methods.py), [looping/modal_stay_solved.py](looping/modal_stay_solved.py), and [modal_eval.py](modal_eval.py): optional Modal wrappers.

## Results

The reference checkpoints were re-evaluated on 2026-09-02 using the frozen 25K-puzzle benchmark, eager CUDA execution, and batch size 256. FP32 and BF16 rows use identical weights. The BF16 late-state CE result reproduced its historical counts exactly. Recheck and margin results are kept in the [research notes](looping/EXPERIMENTS_LOOPING.md#staged-recheck-and-margin).

| Model | 128 iterations | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| **late-state CE only, final, FP32 (recommended)** | 96.29% | **99.12%** | 99.05% | 98.63% |
| released plain-backprop checkpoint, FP32 | 95.70% | 98.89% | 99.03% | 99.02% |
| same late-state CE checkpoint, BF16 | 96.47% | 98.80% | 92.98% | 43.70% |
| same released plain-backprop checkpoint, BF16 | 95.26% | 98.88% | 98.84% | 84.89% |

Precision and compilation can substantially change very long trajectories. Keep execution settings with the score; [numerical checks](release/PRECISION_RESULTS.md) evaluated these effects without changing the weights. FP32 is the default; use `--precision bf16` for the historical arithmetic. `--compiled` enables compiled 16-iteration chunks and is measured separately, not required for the recommended FP32 results. The observed full FP32 evaluation took about 16.5 minutes, versus 12.5 minutes for eager BF16 on H200; these individual runs are not a controlled throughput benchmark.

The model is sudoku-agnostic in the sense that it only assumes a 2D grid: no row, column, or box constraint embedding, just 2D RoPE in attention. Full scaling tables, stability analysis, interventions, and ablations live in [looping/EXPERIMENTS_LOOPING.md](looping/EXPERIMENTS_LOOPING.md) and [iters/EXPERIMENTS_ITERS.md](iters/EXPERIMENTS_ITERS.md).

## Research Notes

- [Looping and late-state training](looping/EXPERIMENTS_LOOPING.md): reliability cohorts, rechecks, margin losses, damping, and reduced training horizons.
- [Iteration scaling](iters/EXPERIMENTS_ITERS.md): the original checkpoint and long-horizon behavior.
- [Recurrent normalization](stabilize/EXPERIMENTS_STABILIZE.md): RMSNorm, caps, and other stabilization experiments.
- [Evolution strategies](es/EXPERIMENTS_ES.md): checkpoint rescue, polish, and training from scratch.
- [Recurrent-state geometry](looping/trajectory_viz/study/README.md): visualization study, controls, and limitations.
- [Release verification](V2_RELEASE_AUDIT.md): artifact provenance and release-preparation status.
- [Numerical sensitivity](release/PRECISION_RESULTS.md) and [burn-in dropout](looping/BURNIN_DROPOUT.md): fixed-checkpoint checks and matched continuation experiments.

## Historical / Archived Code

Older Kaggle and pre-`sudoku-extreme` experiments are preserved for reference, but they are not the current public path:

- `STALE_EXPERIMENTS_DOC.md` - archived chronological experiment log
- `arch/`, `recur/`, `curriculum/`, `misc/` - older experiment families
- `pos_embedding/EXPERIMENTS_POS.md` - 2D RoPE introduction and positional-encoding ablations
- `muon/EXPERIMENTS_MUON.md` - Muon optimizer experiments
- `rrn/RRN_EXPERIMENTS.md` - RRN experiments
- root-level scripts such as `eval_extreme.py`, `eval_only.py`, and `eval_difficulties.py` - archival only
