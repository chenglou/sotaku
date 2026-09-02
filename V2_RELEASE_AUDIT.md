# V2 release audit

Audited 2026-09-02 at commit `42d5739`, branch `codex/viridian-diagnostics`.

Documentation follow-up: the README now leads with late-state CE, describes the actual curriculum pools, names the reference seed and launchers, distinguishes 20K development from 50K reference training, and explains benchmark reuse. The runtime evaluator/resume issues and unpinned Modal environment remain open.

## Verdict

There is a credible v2 to release from the existing experiments. Following the release discussion, late-state cross-entropy is the main README model and recommended release checkpoint; recheck and margin remain research notes. Retain v1 as a reference. The late-state checkpoint uses the same 796,937-parameter architecture and ordinary inference: no recurrent RMSNorm, ES, or inference damping is required.

The repository is not release-ready as currently presented. The public default branch is still v1, the new weights are not published, reproduction defaults need clarification, and the shared evaluator and resume validation have correctness gaps. These need a focused release-preparation pass, not new architecture research.

This audit retrieved existing artifacts and performed local checks. It did not launch GPU jobs, retrain models, change training code, merge branches, create a tag, or publish a release.

## Findings

### P1: The public default branch and release still expose v1

Live GitHub inspection found `master` at `1bbdc32`, with the research branch 80 commits ahead and no divergent commits. The only release is `baseline-lr2e3-checkpoint`, containing `model_baseline_lr2e3.pt`. There are no open pull requests and no GitHub Actions workflows.

Prepare the release on the current research branch or a release-preparation branch, then integrate the reviewed commit into `master` and tag that commit as `v2.0.0`. Keeping the existing default branch name avoids unnecessary migration. Changing the default to a research branch is not required. Preserve the v1 release and attach the v2 weights, configuration, checksums, and evaluation records to the new release.

The difference from `master` includes 500 files, much of it experimental code and analysis artifacts. This audit covers the recommended training, checkpoint, evaluation, and publication paths, not an exhaustive review of every historical experiment.

### P2: The documented curriculum does not match the actual training pools

The audited README described initial rating thresholds of 21+ and 6+. In [`stabilize/exp_testbed_20k.py`](stabilize/exp_testbed_20k.py), line 783 selects entire rating buckets by their lower endpoint. The actual first two pools are 51+ and 11+. The subsequent pools are 1+ and all ratings. The original baseline trainer uses the same bucket-selection behavior. The README and experiment summary now describe the actual pools; code and historical config labels are unchanged.

This does not invalidate the recorded checkpoint scores, but it matters for reproduction. Document the actual historical pools and test their membership. Do not silently change the training data to literal 21+/6+ while describing that as the recipe that produced these weights; that would be a new experiment.

### P2: The quickstart is not the exact reference run, and launch defaults disagree

The simple checkpoint scoring 98.796% at 1024 used seed `20260730` through [`looping/exp_health_methods.py`](looping/exp_health_methods.py). The audited README's `exp_stay_solved.train(...)` command defaulted to `20260724`: the same training method, but not the score-producing seeded run. The historical control with that seed finished at 94.852% on the full 1024 benchmark. The updated README uses the reference preset and seed for its 50K commands and provides separate 20K development commands.

The Python wrapper defaults to 20K steps, whereas [`looping/modal_stay_solved.py`](looping/modal_stay_solved.py), line 109, still defaults to the 10K screen. The README now keeps `--no-screen` explicit for its 20K Modal example. The launcher default itself has not changed.

Expose explicit reference and development presets, seeds, run names, and checkpoint paths. Keep 20K for development and 50K for the released reference recipes. The combined launcher currently hardcodes one private-volume source checkpoint; accept an explicit source path and verify its saved step and configuration.

### P2: Generic evaluation can silently execute the wrong recurrence

[`iters/eval_more_iters.py`](iters/eval_more_iters.py), lines 44-48 and 92-109, instantiates the default architecture and implements its own loop. It does not restore non-weight model settings or call the model's `recurrent_step`. A normalized or capped checkpoint can load successfully, since normalization adds no parameters, and then be evaluated without its training-time constraint. Layer schedules and feedback settings can also be lost.

The two proposed v2 checkpoints are plain four-layer models, so this bug does not invalidate their reported evaluations. The audit confirmed exact agreement between their normal forward and the evaluator's forward at 16 iterations. A separate RMSNorm check reproduced a large output discrepancy when using the generic evaluator.

Use one shared recurrence for training and evaluation, with explicit architecture metadata. Alternatively, restrict the public evaluator to the supported plain model and reject incompatible configurations. Public tensor-only weights load successfully with `weights_only=True`; resumable training checkpoints contain NumPy RNG state and do not load through that same restricted loader. Clearly separate inference artifacts from trusted training-resume artifacts.

### P2: Resume validation misses removed settings

[`checkpoint_utils.py`](checkpoint_utils.py), lines 59-67, checks only keys present in the requested configuration. Removing a previously saved setting therefore escapes validation. The audit reproduced this with the combined checkpoint: removing `late_margin_floor_weight` and `late_margin_floor` was silently accepted.

Validate the union of saved and requested settings, accounting explicitly for supported historical defaults. `load_branch_checkpoint` already uses the stricter comparison. Add a regression test for ordinary resume with a removed loss or model setting.

### P2: The documented environment is not the training environment

The saved runtime logs identify both proposed models as PyTorch `2.10.0+cu128`, CUDA 12.8, driver `580.95.05`. Their Modal image specifies Python 3.11. However, [`requirements.txt`](requirements.txt) pins PyTorch 2.9.1, and [`requirements-modal.txt`](requirements-modal.txt) leaves PyTorch and NumPy unpinned. The README now installs the reference PyTorch build explicitly before the smaller dependency set and warns that this does not pin the remote Modal image. The local audit environment is different again: Python 3.9.6 and PyTorch 2.8.0 on CPU.

Record and pin a tested reference environment. Separate minimal inference dependencies from CUDA training and optional analysis/Modal dependencies. The current general setup includes Linux-specific NVIDIA/Triton packages without stating the platform requirement. Training also requires substantial memory at batch size 2048; late-state gradient accumulation is explicitly unsupported. A tiny checkpoint does not imply that the reference training run fits on a small GPU.

### P2: Benchmark reuse limits the claims the release should make

Training gradients use the training split. However, monitoring, checkpoint selection, and final reporting all draw from the same official test split, and the sampled sets are not disjoint. Reconstructing the index selection on the retained dataset revision found 73 of the control's 1,000 probe puzzles and 77 of the simple checkpoint's probe puzzles in the final 25K benchmark. Their 25K monitoring sets overlap the final benchmark by 2,035 and 2,034 puzzles, respectively. Repeated full-set comparisons also informed experiment decisions.

Report the existing numbers as the repository's balanced development benchmark, not an untouched final holdout or an apples-to-apples comparison with methods using different training data. Save exact evaluation indices and reserve a disjoint validation/test arrangement for future experiments. A genuinely untouched claim needs a new, demonstrably unused evaluation set; changing a random seed alone is not sufficient.

The combined model solves 22 more benchmark puzzles than v1 at 1024, an increase from 98.912% to 99.000%. That is a measured best score, not by itself evidence of a broadly superior method. Paired per-puzzle errors and independent seeds are needed to characterize the improvement. V1 remains stronger at 2048. The combined recipe has only one demonstrated seed; late-state CE has multiple healthy 50K runs, but no guaranteed success rate.

## Audited checkpoints

These results were recovered from the completed Modal evaluation logs, not newly recomputed on the full benchmark during this audit. All use ordinary, undamped inference.

| Checkpoint | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| Simple late-state CE, final | 96.472% | 98.796% | 92.976% | 43.704% |
| Combined CE/recheck/margin, final | 96.256% | 99.000% | 98.456% | 82.172% |
| V1 published checkpoint | 95.3% | 98.912% | 98.8% | Not in the release table |

Use late-state CE as the default training recipe and main checkpoint. Keep the combined checkpoint's results and provenance as research material, not the README's main recommendation. Keep v1 available, including its stronger 2048 result.

### Simple candidate

- Volume: `sudoku-outputs`
- Weights: `looping/model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt`
- SHA-256: `a12508bd32263596b9d87cd5a2f315c0b2d6ded7e66b7f684a05e46397f51198`
- Seed: `20260730`; final saved step: `49999`, meaning 50,000 optimizer updates.
- Resume artifact: `looping/loop_health_standalone_v1_late_state_ce_50k_trial0_checkpoint_step49999.pt`
- Result: `looping/result_loop_health_standalone_v1_late_state_ce_50k_trial0.json`
- Evaluation log: `model_loop_health_standalone_v1_late_state_ce_50k_trial0_eval.log`

### Combined research checkpoint

- Volume: `sudoku-outputs`
- Weights: `looping/model_loop_stay_late_switch_margin_floor5_from39k.pt`
- SHA-256: `b5aa3cda9a770153b977e6df32cb9fb80ba076cedb251d4ac26e2ad1e3c70b77`
- Seed: `20260724`; final saved step: `49999`.
- Source: `looping/loop_stay_control_50k_trial0_checkpoint_step39000.pt`, including optimizer and RNG state.
- Resume artifact: `looping/loop_stay_late_switch_margin_floor5_from39k_checkpoint_step49999.pt`
- Result: `looping/result_loop_stay_late_switch_margin_floor5_from39k.json`
- Evaluation log: `model_loop_stay_late_switch_margin_floor5_from39k_eval.log`

Both local and Modal caches retain dataset revision `58942f96baeb572ca3127e2a9e9c70f330783d6b`. The test split contains 422,786 puzzles; the published benchmark samples 5,000 from each of five difficulty buckets. The training pool is the first 2.7M of 3,831,994 training puzzles. Neither the dataset revision nor a code revision is embedded in the current weight exports; add both to the release manifest.

Describe the training budget precisely: backpropagation spans 16 iterations at a time, but training executes longer trajectories. The simple recipe reaches iteration 528, and the combined recipe reaches 800. The simple method averages 55.68 forward iterations per optimizer batch under its intended sampling distribution, versus 16 for the original recipe. That ratio is not a measured wall-clock training-cost ratio because the additional iterations have no backward pass.

## Release preparation

1. Make a short public inference path and explicit training presets that call the existing verified implementation. Keep historical experiments accessible without requiring users to navigate them.
2. Correct curriculum documentation, seed/default inconsistencies, resume validation, and evaluation configuration handling. Preserve the historical recipe when exporting its checkpoints.
3. Add release manifests containing weight checksums, architecture settings, training recipe and seed, source checkpoint identity, code/data revisions, evaluation indices, precision, and environment.
4. Test installation in a clean supported environment and add CI for the public model, checkpoint loading, resume checks, and a small inference fixture. Core tests should not require a Modal account.
5. Re-evaluate the exact exported bytes on the frozen 25K benchmark with a pinned GPU environment. Evaluate late-state CE and v1 on the same indices and record per-puzzle outcomes at 128/1024/2048/4096. Keep any new holdout results separate from these historical benchmark results; combined-model comparisons remain research.
6. Update the README and release notes, integrate the reviewed commit into `master`, tag `v2.0.0`, and publish the assets. Verify the download-and-evaluate flow from the default branch before announcing.

No new 50K training run is required to establish the identities or historical scores of the proposed release weights. Additional seeds are needed before promoting the combined training recipe as reliably superior.

## Research priorities after the release

The useful next questions are concrete and testable; none needs to be bundled into v2.

- **Separate numerical sensitivity from learned instability.** Test fixed checkpoints across FP32/BF16, eager/compiled execution, and batch sizes on identical puzzles. The project already records different continuations after restoring a checkpoint. This matrix can establish how much of the variation is numerical before introducing another loss.
- **Test the training/inference dropout mismatch.** Detached burn-in runs under `no_grad`, but the model remains in training mode, so dropout is active. Inference disables dropout. Compare the same late-state recipe with and without dropout during detached state preparation, preserving the supervised window and other settings. This is an untested hypothesis in the reviewed current-recipe experiment records, not an assumed improvement; dropout may provide useful regularization.
- **Measure when correct solutions are lost.** In addition to final accuracy, record first-solve iteration, later regressions, and retention across seeds. A 99% snapshot does not establish that further iterations are safe. Use these measurements to decide whether additional late-state supervision is needed.
- **Compare against the strongest existing method before adding complexity.** Prioritize matched-seed late-state CE versus the combined continuation, at equal stated training budgets. Keep RMSNorm, damping, and ES optional; the current evidence does not make them necessary parts of v2.
- **Narrow the geometry claims.** The study did not establish a universal shape, but that is not proof that no shared structure exists. It uses one checkpoint per regime and 20 final puzzles. Its own constraints report notes that direct cross-checkpoint axes were not aligned first. Different hidden coordinate systems can make the same underlying feature fail raw-vector transfer. Prefer functional interventions and held-out alignment tests over stronger conclusions from attractive or dissimilar PCA plots.

The current model embeds the puzzle once; it does not explicitly re-inject the original input on every iteration. An older experiment removed that re-injection, so adding it back should be described as revisiting an existing design choice under the new training recipe, not as a previously unexplored idea.

## Verification performed

- `python -m unittest discover -v`: 115 tests passed in the existing local environment.
- Both new weight files load with `weights_only=True`, have 796,937 parameters, and match every tensor in their respective final resumable checkpoints.
- Both resume artifacts contain optimizer state and Python, NumPy, CPU Torch, and CUDA RNG state.
- The v1 and two v2 models produced finite outputs through 1024 iterations on a five-puzzle CPU smoke fixture, with one puzzle per difficulty bucket. This is a load/inference check, not a benchmark estimate.
- The public evaluator's 16-step forward agrees exactly with the model forward for all three plain checkpoints. The normalization-bypass and removed-config resume defects were reproduced separately.
- Live default branch, branch divergence, releases/assets, open pull requests, and CI workflow state were checked. A limited tracked-file scan found no common private-key or API-token patterns; this is not an exhaustive security audit.
- `git diff --check` passed before this report. The user's existing untracked `temp-side-convo.txt` was left untouched.

Still unverified: a clean dependency installation, fresh full-set GPU evaluation of the release exports, and the eventual public release download path. Artifact downloads and the local smoke-check script are retained under `/tmp/sotaku-v2-audit-20260902.4jekvM` for the release-preparation pass.
