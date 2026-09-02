# Sotaku V2 Checkpoint

The reference checkpoint uses training on later iterations: the same four-layer looped transformer, with ordinary cross-entropy applied to the next 16 iterations after running an initial 32, 64, 128, 256, or 512 iterations without gradients on 20% of training batches. The other 80% train on iterations 1-16. No auxiliary loss, normalization, ES, or inference damping is required.

The unchanged final weights from seed `20260730`, trained for 50,000 optimizer updates, solve **99.116% at 1024 iterations** and **98.632% at 4096** on the repository's frozen 25K-puzzle benchmark using eager FP32 inference. See the [complete numerical results](../PRECISION_RESULTS.md). The benchmark has been reused for development and selection; it is not an untouched holdout.

## Files

- `model_late_state_ce.pt`: tensor-only inference weights, SHA-256 `a12508bd32263596b9d87cd5a2f315c0b2d6ded7e66b7f684a05e46397f51198`.
- `model_late_state_ce.pt.json`: model settings, training configuration, source checksums, and reference evaluation.
- `validation_records_20260902.zip`: all 34 evaluated conditions, per-puzzle predictions, benchmark indices, training summaries, and source snapshots. Verify the extracted records without a GPU using the [record verification instructions](../validation/README.md).
- `validation_records_20260902.checksums.json`: SHA-256 and byte counts for the weights, manifest, and validation archive. Large assets are excluded from Git.

Download the assets from [v2.0.0](https://github.com/chenglou/sotaku/releases/tag/v2.0.0). Keep the weights and their `.pt.json` manifest together; the public inference tools verify the weight checksum and restore the recorded model settings.

## Download And Evaluate

```sh
source venv/bin/activate
gh release download v2.0.0 --repo chenglou/sotaku --pattern 'model_late_state_ce.pt*'
python -m iters.eval_more_iters model_late_state_ce.pt \
  --benchmark release/benchmark_25k.json --precision fp32 --device cuda \
  --batch-size 256 --iters 128 1024 2048 4096
```

Use CUDA, batch size 256, and the pinned environment for the reported results. The adjacent manifest is required. FP32 is the default; BF16 and compilation remain explicit alternatives because they can change long trajectories even with identical weights.
