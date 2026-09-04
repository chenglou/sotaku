# Selecting Training Windows

Protocol fixed before the runs. This tests the [suggestion to choose training iterations by confidence](https://x.com/LordoftheMounts/status/2095379923452203074), not search or answer selection at inference. The released model and inference defaults are unchanged.

## Comparison

Nine fresh runs: random, highest-confidence, and latest-window selection, each with seeds `20260904`, `20260905`, and `20260906`. Every run uses the same four-block, 796,937-parameter model, 20K optimizer updates, batch size 2048, and 2.7M-puzzle training pool. The learning rate and curriculum match the existing 20K recipe. Cross-entropy is averaged across 16 iterations. There is no auxiliary loss, added normalization, ES, or inference damping.

On 80% of batches, train iterations 1-16 as usual. On the other 20%, all three variants scan to iteration 528 without gradients and consider the five windows starting after iterations 32, 64, 128, 256, and 512. For example, start 128 means loss on iterations 129-144. Choose one window for the entire batch, matching the current trainer's selection granularity:

- `random`: choose uniformly among those five starts.
- `confidence`: choose the largest mean maximum digit probability over originally blank cells and all 16 iterations. Compute the probabilities in FP32 to reduce rounding ties; exact ties select the earlier window. No answer labels enter selection.
- `latest`: always choose start 512.

Restart from the selected window's saved hidden state and prediction feedback, then replay 16 iterations with gradients. The scan and replay both retain dropout. Restore the RNG state from before the chosen scan window for replay, then restore the end-of-scan RNG state before the next batch. This keeps future random-number consumption independent of the selected location. Compilation can still make gradient-free and gradient-tracked arithmetic differ; log replay confidence separately rather than claiming bit-identical trajectories.

The random variant uses the current selection distribution but pays the same full scan/replay cost as the other variants. It is therefore a matched experimental control, not an implementation-speed comparison against the existing short-burn-in trainer. Independent sampling streams keep puzzle batches, late-batch decisions, and unused random candidate draws identical across each three-seed group. All three variants record these streams and verify their rolling digest. The model parameters are fixed during each scan and replay and updated once afterward.

## Evaluation And Decision

The [machine-readable protocol](protocol.json) fixes the criteria. Primary results are the final 20K checkpoints on the frozen, reused 25K sudoku-extreme benchmark at 1024 inference iterations. Also evaluate 128, 2048, and 4096, with ordinary eager FP32 inference and TF32 matmul disabled. Track each puzzle's first correct answer, later losses of correctness, and endpoint gains/losses. This benchmark has been used for development; it is not an untouched holdout.

Call confidence selection promising only if all runs finish with finite results, its paired mean 1024 accuracy exceeds random by at least one percentage point, at least two of three seed pairs improve, and its mean 4096 accuracy does not fall below random. Report the latest-window control separately: matching that control does not establish a benefit from adapting to confidence. Three seeds do not establish precise training-success probabilities. A promising result requires a separate 50K confirmation before changing the public recommendation.

Save and evaluate the best monitoring checkpoint as a secondary result, without replacing an unfavorable final result. Report each seed, mean/minimum monitoring accuracy from steps 12K-20K, selection frequencies, actual training time, and estimated recurrent work. Record selected and unselected windows' cell accuracy and confidently wrong predictions for diagnosis only; these answer-dependent measurements cannot affect the selector.

## Implementation And Runs

The experiment reuses the public model and evaluator, plus the preceding study's frozen prepared training/monitoring arrays. Data checksums are pinned in the protocol. New artifacts live only under `/outputs/window_selection_v1_20260904/` on the `sudoku-outputs` Modal volume. Every resumable checkpoint includes model, optimizer, step, configuration, source hashes, sampler states, dropout RNG, and monitoring history. Selected weights are saved immediately when the monitoring score improves.

Before training, CPU tests verify exact model/loss equivalence, dropout replay, paired sample streams, and interrupted-versus-uninterrupted training. A full-batch CUDA preflight verifies finite gradients, compiled scan/replay behavior, export loading, and restoration of populated optimizer and RNG state. Each separate detached invocation spawns one worker. A training worker evaluates its own final and selected checkpoints after committing the training results; a retry resumes training or reuses verified completed artifacts instead of starting another run.

```sh
source venv/bin/activate
python -m unittest looping.window_selection.test_selection -v
modal run --detach looping/window_selection/modal_run.py --action smoke
modal run --detach looping/window_selection/modal_run.py --action train --selector random --seed 20260904
```

Use a separate invocation for each selector/seed. [jobs.json](jobs.json) records actual launches; no other jobs are implied by the commands above.
