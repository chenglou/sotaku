# Hyperloop: 50K Confirmation

Six fresh 50K runs compare the current one-state recipe with four-state Hyperloop. The [completed 20K study](../hyperloop/RESULTS.md) improved 1024-iteration accuracy in all three pairs, but both architectures still had a deep-iteration failure. These runs check whether the improvement survives the full training schedule. The public release and defaults stay unchanged.

## Design

`baseline` and `gated_four` each use paired seeds 20260910, 20260911, and 20260912. These seeds are new, not selected from earlier results. Every run starts from fresh weights, not a completed 20K checkpoint whose learning-rate schedule has ended.

The models and training calculations are unchanged: four shared transformer blocks, width 128, dropout 0.1; 796,937 parameters for baseline and 803,096 for Hyperloop. Both use the same 2.7M-puzzle pool, batch size 2048, AdamW at 2e-3, and 16-iteration averaged cross-entropy. On 20% of batches, advance 32/64/128/256/512 iterations without gradients before supervising the next 16. No extra losses, carried-state normalization, ES, damping, or inference search.

The 50K schedule has 1,400 warmup updates followed by cosine decay. Sample ratings 51+ until 10K, 11+ until 20K, 1+ until 30K, then all ratings until 50K. These are the public recipe's effective pools: its older labels say 21+ and 6+, but whole-bucket selection produces 51+ and 11+. Puzzle/depth samplers and initial base weights match within each pair; different compiled graphs need not produce identical dropout or floating-point trajectories.

Evaluate the fixed 1K monitoring sample every 1K updates. Save the earliest checkpoint with the highest 1024-iteration monitoring score immediately. After training, evaluate both final and selected checkpoints on all 25K benchmark puzzles at 16, 128, 512, 1024, 2048, and 4096 iterations using eager FP32 with TF32 disabled. Record predictions and solution regressions without altering inference. These samples are reused development data, not an untouched holdout.

## Criteria Recorded Before Training

Final checkpoints are primary. Report every seed, learning curves, mean/minimum 1024 monitoring accuracy from 30K onward, training time, and full-set accuracy. A reliable final checkpoint has at least 90% at 1024, at least 85% at 4096, and no more than a five-percentage-point decrease between them. This is a failure-screening threshold, not a SOTA target.

Keep the original paired comparison criteria: either improve mean 1024 accuracy by at least 0.5 percentage points without lowering mean 4096 accuracy, with at least two positive 1024 pairs; or improve mean 4096 accuracy by at least five points while losing at most 0.5 points at 1024, with at least two positive 4096 pairs. All six runs must complete with finite results. A release recommendation additionally requires all three Hyperloop finals to meet the reliability threshold; no automatic promotion.

## Verification And Operation

`protocol.json` freezes settings and data identities. The original 20K sources remain unchanged. The separate trainer and evaluator differ only in protocol imports and export identity, enforced by an exact source comparison. This small duplication keeps archived experiments runnable without changing their hashes.

Tests cover the effective 50K curriculum, paired sampling across transitions, exact interrupted/resumed training, selected-weight recovery, mismatched artifacts, and independent scoring of saved predictions. The new GPU check uses batch size 8, a 512-iteration prefix, populated optimizer/RNG restoration, and FP32 inference through 4096. It supplements the existing compiled batch-2048 preflight, not another full-batch test. Reusing that evidence requires unchanged original source hashes, numerical code, model, batch size, and training depths. Training also checks its source hashes against the newly passed GPU check.

Each detached H200 job runs training and both full evaluations, with optimizer/RNG/sampler checkpoints, retries, and volume commits. The upload uses an explicit file allowlist. Outputs stay separate under `/outputs/hyperloop_50k_v1_20260905/` on `sudoku-outputs`.

```bash
source venv/bin/activate
python -m unittest looping.hyperloop_50k.test_confirmation
modal run --detach -m looping.hyperloop_50k.modal_run --action smoke
# After the GPU check passes, use separate invocations for each arm/seed:
modal run --detach -m looping.hyperloop_50k.modal_run --action train --arm baseline --seed 20260910
modal run --detach -m looping.hyperloop_50k.modal_run --action train --arm gated_four --seed 20260910
# Repeat those two commands separately for seeds 20260911 and 20260912.
```

## After Confirmation

If results hold, compare shorter schedules with architecture fixed, then narrower models with schedule fixed. Measure actual training and inference time as well as parameters and accuracy. Test smaller datasets separately from fewer optimizer updates. Those reductions are not launched yet; a high-scoring checkpoint alone does not establish benchmark saturation.
