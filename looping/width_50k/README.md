# Width 160: 50K Confirmation

Three fresh width-160 runs tested the full 50K schedule after [the 20K screen](../width/RESULTS.md). They started from new weights, not the completed 20K checkpoints.

**Completed September 6:** all three runs finished 50K updates and both full 25K-puzzle evaluations. Final models averaged 99.02% at 1024 and 98.62% at 4096, and all three met the stability criterion. The study passed its predeclared confirmation criteria. See [results](RESULTS.md) and [job records](jobs.json). The release and defaults remain unchanged.

Compare seeds 20260910, 20260911, and 20260912 against all three completed width-128 controls from the 50K Hyperloop study. The controls averaged 98.44% at 1024 and 96.12% at 4096; two of three met the stability criterion. [Frozen reference records](reference.json) retain every control, including the unsuccessful one. These seeds were not used in the width-160 20K screen. Widening changes both recurrent storage and model capacity, not only working memory.

## Fixed Recipe

The model stays at width 160, four shared transformer blocks, four heads, feedforward width 640, dropout 0.1, and 1,241,929 parameters. Use the same 2.7M training puzzles, batch 2048, AdamW at 0.002, and ordinary cross-entropy averaged over 16 iterations. On 20% of batches, first run 32/64/128/256/512 iterations without gradients. No extra losses, recurrent normalization, ES, damping, or inference search.

Start from fresh weights with 1,400 warmup updates and cosine decay over exactly 50K updates. Sample ratings 51+ through 10K, 11+ through 20K, 1+ through 30K, then all ratings. These are the same effective curriculum pools as the archived 50K controls. Exact puzzle/depth sampling and iteration counts must match each paired control; initial weights and dropout draws need not match across widths.

Monitor the fixed 1K sample every 1K updates and save the earliest best 1024 checkpoint immediately. Each worker automatically evaluates both final and selected checkpoints on all 25K benchmark puzzles at 16, 128, 512, 1024, 2048, and 4096 iterations using eager FP32 with TF32 matmul disabled. The reused benchmark is not an untouched holdout.

## Criteria Recorded Before Training

Final models are primary. Report all seeds, paired differences, accuracy averages, the mean/minimum 1024 monitoring score from 30K onward, and actual training/evaluation time. Count every failed or nonfinite run. A stable final model must score at least 90% at 1024, at least 85% at 4096, and lose no more than five percentage points between them.

Keep the 20K comparison criteria: improve mean 1024 by at least 0.5 points without reducing mean 4096, with at least two improved 1024 pairs; or improve mean 4096 by at least five points while losing at most 0.5 at 1024, with at least two improved 4096 pairs. A recommendation additionally requires all three width-160 finals to meet the stability criterion. No automatic promotion, and report parameter/time cost even when accuracy improves.

## Verification And Operation

The separate trainer/evaluator change only study imports and export identity; exact source checks enforce this. The unchanged width-160 compiled batch-2048 GPU check supplies full-batch evidence, verified by all source hashes and matching architecture, batch size, and iteration counts. A new batch-8 CUDA check tests the 50K configuration, a 512-iteration gradient-free prefix, populated optimizer/RNG restoration, and export/reload with FP32 inference through 4096. CPU tests exercise the 50K schedule, paired sampling at curriculum transitions, exact interrupted/resumed training, selected-weight recovery, and rejection of wrong schedules or evaluation artifacts.

Each H200 worker has a 24-hour timeout, retries, atomic checkpoints, and volume commits. The launch uses an explicit source allowlist and one spawn per separate detached invocation. Outputs are isolated under `/outputs/baseline_width_50k_v1_20260906/` on `sudoku-outputs`. [Job records](jobs.json) track the preflight and runs. Do not duplicate an existing launch.

```sh
source venv/bin/activate
python -m unittest looping.width_50k.test_confirmation
modal run --detach looping/width_50k/modal_run.py --action smoke
# After the new GPU check passes, run each command separately:
modal run --detach looping/width_50k/modal_run.py --action train --arm width160 --seed 20260910
modal run --detach looping/width_50k/modal_run.py --action train --arm width160 --seed 20260911
modal run --detach looping/width_50k/modal_run.py --action train --arm width160 --seed 20260912
```
