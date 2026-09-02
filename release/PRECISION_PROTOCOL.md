# Release Verification And Precision Checks

Recorded 2026-09-02 before the evaluations. These checks do not train or alter either reference model.

Results and the resulting FP32 inference default are documented in [PRECISION_RESULTS.md](PRECISION_RESULTS.md).

## Full Benchmark

Evaluate the exact exported later-iteration training weights and the published v1 weights on the historical balanced 25K benchmark at 128, 1024, 2048, and 4096 iterations. Use eager CUDA bfloat16, batch size 256, ordinary undamped recurrence, and PyTorch 2.10.0+cu128 on H200. Verify equality against the original model forward before the full run. Save model hashes, complete model settings, runtime package/driver records, exact puzzle indices, and per-puzzle predictions.

The frozen row-content checksum is `697d79d8494d7904d7ecf5638f3f0e47b43c8f66c6b4381022d0457a469d61bc`. The dataset revision and original sampling order are recorded in `benchmark_25k.json`. This benchmark has been reused during development; it is not a new holdout.

## Numerical Sensitivity

Use the first 200 puzzles in each frozen benchmark bucket, totaling 1,000, chosen without inspecting model outputs. For each reference model, test both FP32 and BF16, eager and compiled execution, and batch sizes 32 and 256. FP32 uses `matmul_precision=highest` to disable TF32 matmul; BF16 retains the historical `high` setting. Compiled inference uses full-graph 16-iteration chunks, not a single 4096-iteration compiled graph.

Report scores and paired puzzle disagreements relative to BF16/eager/256. Do not select the most favorable arithmetic as a new training result. Any significant difference qualifies how the reference score should be reproduced.

Separately record every iteration in eager BF16 and FP32 at batch size 256: first solved iteration, later regressions, and whether a board stayed solved. Assert that observation does not change endpoint predictions. Keeping this separate avoids changing the compiled graph merely to inspect its intermediate states.

## Full-Set Confirmation

The first completed compiled condition changed later-iteration training's 4096-iteration result from 42.2% to 93.4% on the fixed 1K sample. Following that finding, evaluate both existing reference checkpoints on the full frozen 25K in BF16/compiled and FP32/eager, always at batch size 256. Keep these execution modes separate from the historical eager-BF16 score; no weights are selected or changed for this follow-up.
