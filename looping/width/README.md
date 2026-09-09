# Baseline Width

Test whether modestly widening the baseline improves accuracy and long-iteration stability. Six fresh 20K runs use widths 160 and 192, three seeds each. Compare against all three completed width-128 controls from the [20K Hyperloop study](../hyperloop/RESULTS.md), with the same seeds 20260907-20260909. [Job records](jobs.json) track the detached workers.

**Completed, verified September 6:** all six runs finished 20K updates and both full 25K-puzzle evaluations. Width 160 improved average accuracy and all three final models met the stability criterion; width 192 scored higher at 1024 but still had a severe collapse at 4096. See [results](RESULTS.md) and the subsequent [50K width-160 confirmation](../width_50k/RESULTS.md). Defaults remain unchanged.

The single recurrent state increases from 128 to 160 or 192 numbers per cell. Keep four transformer blocks, four attention heads, dropout 0.1, and the feedforward width at four times the hidden width. This also increases parameters and arithmetic; it does not isolate extra working memory from greater model capacity. Positional encoding uses the same construction, with frequencies determined by head width.

| Hidden width | Parameters | Relative to baseline |
|---|---:|---:|
| 128 | 796,937 | 1.00x |
| 160 | 1,241,929 | 1.56x |
| 192 | 1,785,225 | 2.24x |

The training recipe stays fixed: 2.7M training puzzles, batch size 2048, AdamW at 0.002, 560 warmup updates then cosine decay, the same 20K curriculum, and 16 supervised iterations. On 20% of batches, first run 32/64/128/256/512 iterations without gradients. Use the original cross-entropy and ordinary FP32 inference. The complete configuration is in [protocol.json](protocol.json).

## Comparisons Recorded Before Training

Final checkpoints are primary. Evaluate final and selected checkpoints on all 25K benchmark puzzles at 16, 128, 512, 1024, 2048, and 4096 iterations, using eager FP32 with TF32 matmul disabled. Select the secondary checkpoint by the earliest maximum 1024 score on the fixed 1K monitoring sample. The benchmark was reused during development and is not an untouched holdout.

Report all three seeds for each width, paired score differences, final means, the 1024 monitoring mean/minimum from step 12K onward, and actual training/evaluation time. Count failures explicitly. A stable final model must reach at least 90% at 1024, at least 85% at 4096, and lose no more than five percentage points between them.

A width is promising if it improves mean 1024 accuracy by at least 0.5 percentage points without lowering mean 4096 accuracy, with at least two improved pairs; or improves mean 4096 by at least five points while losing at most 0.5 points at 1024, with at least two improved pairs. Report parameter and compute costs alongside accuracy. These are screening runs, not an automatic release change; confirm a promising width at 50K separately.

## Verification

Reuse the existing recurrence, transformer layer, loss, and positional-table builder. Tests require exact equality at width 128 for initialization, outputs, dropout, losses, and gradients, including a gradient-free prefix. The new trainer/evaluator are copies differing only in study imports and artifact type; an exact source comparison enforces that restriction. Archived studies remain unchanged.

[reference.json](reference.json) records all three original controls, their source identities, sampling digests, and full evaluations. Refuse to run if their sources or matching training settings change. A completed wider run must match its control's exact puzzle/depth sampling digest and iteration counts. Different shapes have different initial weights and may use different GPU kernels; paired seeds do not imply identical dropout draws across widths.

Each width must pass a new full-batch compiled H200 check with 16 supervised iterations after a 512-iteration gradient-free prefix, finite gradients, populated optimizer/RNG restoration, and export/reload plus FP32 inference through 4096. A separate detached invocation runs each preflight and each training job. Retries resume model, optimizer, schedule, sampler, and RNG state. Source/config checks reject mismatched checkpoints. Completed workers automatically evaluate both saved checkpoints.

The checks passed at [width 160](results/preflight_width160.json) and [width 192](results/preflight_width192.json), with peak allocated GPU memory of 62.95 and 75.54 GiB. Startup compilation can be slow: the checks took 12.1 and 49.3 minutes. Nonblocking stack inspection located the longer wait in PyTorch's compiler metadata-update pass; no compiler flags or numerical code were changed. These setup times are separate from optimizer-update time recorded by the trainer.

## Running

```sh
source venv/bin/activate
python -m unittest looping.width.test_width
modal run --detach looping/width/modal_run.py --action smoke --arm width160
modal run --detach looping/width/modal_run.py --action smoke --arm width192
# After each width passes, launch each arm/seed in a separate invocation:
modal run --detach looping/width/modal_run.py --action train --arm width160 --seed 20260907
```

Repeat the training command separately for all six arm/seed pairs. Outputs use `/outputs/baseline_width_v1_20260905/` on `sudoku-outputs`. Use the file-path launcher shown above. Do not rerun an already-launched job merely because its detached client reports completion.
