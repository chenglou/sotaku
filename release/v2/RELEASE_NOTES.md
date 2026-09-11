# Sotaku v2

An 800K-parameter looped transformer that solves **99.12%** of our 25,000-puzzle Sudoku benchmark. Train on later iterations, then use FP32 inference.

## Model And Training

The model has 796,937 parameters and four shared transformer blocks. Training uses ordinary cross-entropy over 16 iterations. On 20% of batches, it first runs 32, 64, 128, 256, or 512 iterations without gradients, then trains on the next 16. The other 80% use iterations 1-16.

The released final checkpoint used 50,000 optimizer steps, batch size 2048, a 2.7M-puzzle training pool, and random seed `20260730`. The repository defaults to 20K steps for development and provides the 50K reference preset.

## Results

Ordinary eager FP32 inference, H200, PyTorch 2.10.0+cu128, batch size 256:

| Inference iterations | Solved / 25,000 | Accuracy |
|---:|---:|---:|
| 128 | 24,072 | 96.288% |
| 1024 | 24,779 | **99.116%** |
| 2048 | 24,762 | 99.048% |
| 4096 | 24,658 | 98.632% |

The balanced benchmark uses 5,000 puzzles from each difficulty bucket of `sapientinc/sudoku-extreme`. It has been reused for development and model selection, not reserved as an untouched final holdout. These scores describe the released checkpoint; independent retraining is not guaranteed to reproduce them exactly.

## Download

After [setting up the repository](https://github.com/chenglou/sotaku/tree/v2.0.0#setup):

```sh
gh release download v2.0.0 --repo chenglou/sotaku --pattern 'model_late_state_ce.pt*'

python solve.py model_late_state_ce.pt \
  '53..7.... 6..195... .98....6. 8...6...3 4..8.3..1 7...2...6 .6....28. ...419..5 ....8..79'

python -m iters.eval_more_iters model_late_state_ce.pt \
  --benchmark release/benchmark_25k.json --precision fp32 --device cuda \
  --batch-size 256 --iters 128 1024 2048 4096
```

Keep `model_late_state_ce.pt.json` beside the weights. The manifest records model settings, training configuration, and the reference evaluation; inference verifies the weight checksum. Single-puzzle inference also works on CPU, without downloading the dataset.

The release includes the tensor-only weights, manifest, a validation archive containing per-puzzle predictions and environment records, and a checksum file. [Validation instructions](https://github.com/chenglou/sotaku/blob/v2.0.0/release/validation/README.md) explain how to independently recompute the recorded scores without a GPU.
