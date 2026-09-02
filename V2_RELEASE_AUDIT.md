# V2 Release Audit

Initial audit: 2026-09-02 at `42d5739`, on `codex/viridian-diagnostics`. This document includes the subsequent release-preparation work. Local `master` has since been fast-forwarded to `ec3febe`, incorporating that work. The updated `master` has not been pushed; no v2 tag, release publication, or GitHub default-branch change has been performed.

## Recommendation

For v2, train on later iterations and use ordinary FP32 inference. The architecture remains the same 796,937-parameter looped transformer. The recipe changes which recurrent states receive the ordinary cross-entropy loss; it adds no normalization, auxiliary loss, ES, or inference damping. Keep the published v1 checkpoint available, including its stronger result at 4096 in FP32. Additional supervised windows and margin penalties remain research material.

The release engineering defects identified below have been addressed. Numerical-sensitivity checks and four matched dropout continuations are complete. The reference weights are unchanged; publication is still pending.

## Verification Status

| Area | Result |
|---|---|
| Clean installation | Python 3.11.9, PyTorch 2.10.0; 136 unit tests passed; dependency consistency checked |
| GPU training mechanics | Full four-layer model, 16 supervised iterations; compiled CUDA save/resume and dropout-off burn-in passed |
| Inference equivalence | Exact eager logits versus the historical forward at 16, 128, and 1024 iterations for both reference models: eight CUDA BF16 puzzles and one CPU FP32 fixture |
| Full benchmark | Historical BF16 counts for the model trained on later iterations reproduced exactly; FP32 improves the same checkpoint to 99.116% at 1024 and 98.632% at 4096 |
| Numerical sensitivity | Completed both fixed-weight matrices and every-iteration solution-retention checks; see [results](release/PRECISION_RESULTS.md) |
| Dropout experiment | Four matched 4K continuations and eight full evaluations completed; neither dropout-off seed meets the extension rule. Keep dropout on; see [results](looping/BURNIN_DROPOUT.md) |
| Retained records | 34 evaluation conditions independently recomputed from saved predictions and pinned puzzle data, then verified again after extracting the release archive |
| Public release | [Weights, manifest, and validation archive](release/v2/README.md) prepared locally; public download flow cannot be verified until publication |

The GPU smoke fixture uses five copies of an almost-filled board. It verifies optimization, saved optimizer state, resumption, gradients, and module modes, not Sudoku accuracy. Results are retained in [release/validation](release/validation).

## Fresh Benchmark

Same frozen 25K puzzles, PyTorch 2.10.0+cu128, H200, eager execution, batch size 256, ordinary undamped recurrence:

| Checkpoint | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| Training on later iterations, final, FP32 | 96.288% | 99.116% | 99.048% | 98.632% |
| Published v1, FP32 | 95.704% | 98.888% | 99.028% | 99.020% |
| Same later-iteration training weights, BF16 | 96.472% | 98.796% | 92.976% | 43.704% |
| Same published v1 weights, BF16 | 95.256% | 98.876% | 98.844% | 84.888% |

In BF16, the later-iteration training checkpoint matches its historical counts exactly: 24,118 / 24,699 / 23,244 / 10,926 solved. V1's fresh BF16 1024 count is 24,719, versus the historical 24,728. The difference is not evidence that its weights changed; their checksum is identical. Do not attribute that nine-puzzle historical difference to a particular runtime setting without a controlled comparison.

Changing only arithmetic largely removes the selected later-iteration training checkpoint's 4096 deterioration. FP32 is now the default for public evaluation, single-puzzle inference, and the Modal evaluator; BF16 remains an explicit option for historical reproduction. This does not prove that every older checkpoint with poor accuracy had the same cause, or that further FP32 iterations can never fail. Training remains BF16 and its original dropout setting is unchanged.

Each evaluation retains exact row indices, predictions and correctness per puzzle, model settings and checksum, runtime packages and driver, and source-file hashes. The canonical row-content SHA-256 is `697d79d8494d7904d7ecf5638f3f0e47b43c8f66c6b4381022d0457a469d61bc`.

## Engineering Fixes

- **Resume validation:** compares added, changed, and removed configuration keys, with explicit historical defaults. Branches verify the source step and seed. The removed-margin-setting regression has a test.
- **Inference correctness:** the evaluator calls the model's actual `recurrent_step` and restores non-weight settings from a checksummed manifest. The old independent loop could silently omit recurrent normalization, capping, or a layer schedule.
- **Artifact safety:** public weights are tensor-only and use `weights_only=True`. Training checkpoints contain pickled optimizer/RNG state and require explicit trust when exporting. New exports record the selected step; final resumable checkpoints include the final monitoring history.
- **Public entrypoints:** `train.py` exposes 20K development and 50K reference presets with explicit seed and run name. `solve.py` accepts one puzzle. The Modal development launcher now also defaults to 20K; 10K requires `--screen`. Late-switch runs accept and validate an explicit source checkpoint.
- **Dataset identity:** training and evaluation pin revision `58942f96baeb572ca3127e2a9e9c70f330783d6b`. The 25K benchmark indices and row-content hash are frozen.
- **Curriculum documentation:** the original code selects whole rating buckets. Its first pools are 51+ and 11+, not literal 21+ and 6+. Documentation now describes those actual pools, and a regression test preserves historical behavior.
- **Environment:** the core requirements and CUDA build are pinned, the previous environment snapshot is retained separately, and evaluations record transitive package versions. A CPU GitHub Actions workflow exercises the public model and checkpoint tests without a Modal account.

The shared training math and default dropout behavior are preserved. Additional metadata and explicit burn-in-dropout controls do not silently change the reference recipe.

## Reference Artifact

The prepared tensor-only file is `release/v2/model_late_state_ce.pt`, with its adjacent [manifest](release/v2/model_late_state_ce.pt.json). Exporting verified every tensor against the final resumable checkpoint and preserved the original weight-file bytes.

- Seed: `20260730`; last step `49999`, meaning 50,000 optimizer updates.
- Source volume: `sudoku-outputs`.
- Original weights: `looping/model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt`.
- Weight SHA-256: `a12508bd32263596b9d87cd5a2f315c0b2d6ded7e66b7f684a05e46397f51198`.
- Resume source: `looping/loop_health_standalone_v1_late_state_ce_50k_trial0_checkpoint_step49999.pt`.
- Result: `looping/result_loop_health_standalone_v1_late_state_ce_50k_trial0.json`.

The historical checkpoint did not record its exact training commit. The manifest records that as unknown, rather than substituting the current commit. The retained configuration and historical implementation defaults establish its settings; new runs also record source-file hashes and their runtime.

Training uses the first 2.7M of 3,831,994 training puzzles, batch size 2048, AdamW at an initial LR of 2e-3, 1,400 warmup steps, and cosine decay. On 20% of batches, the model first runs 32/64/128/256/512 iterations without gradients, then trains through the next 16; the other 80% use ordinary iterations 1-16. These initial gradient-free iterations are called burn-in. A training example can therefore reach iteration 528, although gradients span only the final 16 iterations. The expected forward depth is 55.68 iterations per batch, but that is not a measured wall-clock cost ratio because burn-in has no backward pass.

## Benchmark Limits

Training gradients use the training split. Monitoring, checkpoint selection, and final reporting reuse the official test split, and the samples overlap. The two audited groups of runs had 73 and 77 puzzles from their 1K monitoring samples in the final 25K benchmark; their larger monitoring sets overlapped by 2,035 and 2,034 puzzles. Full-set results also informed experiment decisions.

Call these results the repository's balanced development benchmark, not an untouched final holdout. Future generalization claims need demonstrably unused data. Training on later iterations has produced multiple 50K runs above the 90% monitoring threshold at 1024 iterations, but no guaranteed success rate. The 20K schedule is the development default for speed; it does not rule out later improvements or accuracy losses. A smaller training dataset has not been tested with the current recipe.

## Preserved Research

The combined experiment trained on later iterations, then added a second supervised window and minimum-margin penalty at step 39K. Its checkpoint remains documented in [the looping notes](looping/EXPERIMENTS_LOOPING.md#additional-training-window-and-margin-penalty). Its historical full profile is 96.256 / 99.000 / 98.456 / 82.172%, without damping. Its weights are `looping/model_loop_stay_late_switch_margin_floor5_from39k.pt`, SHA-256 `b5aa3cda9a770153b977e6df32cb9fb80ba076cedb251d4ac26e2ad1e3c70b77`, from random seed `20260724` through step `49999`. That selected run is not sufficient evidence of a reliably superior recipe.

The geometry study did not establish a universal shape. It also did not prove that none exists: it used one checkpoint per regime, 20 final puzzles, and unaligned hidden coordinates for some transfer tests. Prefer controlled interventions to stronger conclusions from PCA appearance. Re-introducing the original puzzle on every loop would revisit an older design choice; the current model embeds it only once.

## Publication Steps

The initial GitHub audit found `master` at `1bbdc32`, with the research branch 80 commits ahead and none behind. Only the `baseline-lr2e3-checkpoint` release was published. Keep `master` as the default; changing the default to a research branch is unnecessary.

1. Review the committed release preparation, final numerical report, and retained evaluation records.
2. Local integration and documentation review are complete: `master` was fast-forwarded to `ec3febe`, followed by the terminology cleanup. Push the reviewed commits before tagging. The research changes include about 500 files; this audit covers the supported training, inference, checkpoint, and release paths, not every archived experiment.
3. Tag the integrated commit, publish the tensor-only weights, manifest, checksums, and evaluation records, and retain the v1 release.
4. Test downloading and evaluating the published assets from the public default branch before announcing.

No fresh 50K run is required to identify or verify the existing release candidate. The dropout continuations test a separate hypothesis and must not be presented as part of the original checkpoint's training.
