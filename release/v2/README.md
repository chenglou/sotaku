# Prepared V2 Assets

The reference checkpoint is late-state CE only: the same four-layer looped transformer, with ordinary cross-entropy applied after longer detached preparation on 20% of training batches. No auxiliary loss, normalization, ES, or inference damping is required.

The unchanged final weights from seed `20260730`, trained for 50,000 optimizer updates, solve **99.116% at 1024 iterations** and **98.632% at 4096** on the repository's frozen 25K-puzzle benchmark using eager FP32 inference. See the [complete numerical results](../PRECISION_RESULTS.md). The benchmark has been reused for development and selection; it is not an untouched holdout.

## Files

- `model_late_state_ce.pt`: tensor-only inference weights, SHA-256 `a12508bd32263596b9d87cd5a2f315c0b2d6ded7e66b7f684a05e46397f51198`.
- `model_late_state_ce.pt.json`: model settings, training configuration, source checksums, and reference evaluation.
- The validation archive and its adjacent `.checksums.json`, generated after the checks finish, contain the evaluation records and source snapshots. Large assets are excluded from Git.

Publication is separate from preparation. No v2 tag, public release, merge, or default-branch change has been made by this audit. Keep the v1 release available.

## Local Verification

```sh
source venv/bin/activate
python -m iters.eval_more_iters release/v2/model_late_state_ce.pt \
  --benchmark release/benchmark_25k.json --precision fp32 --iters 128 1024 2048 4096
```

Use CUDA, batch size 256, and the pinned environment for the reported results. The adjacent manifest is required. FP32 is the default; BF16 and compilation remain explicit alternatives because they can change long trajectories even with identical weights.
