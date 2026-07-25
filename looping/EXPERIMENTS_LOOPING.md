# Looped Dynamics Experiments (July 2026)

This folder contains controlled follow-ups to two July 2026 results: the layer-loop schedule in [*Loop the Loopies!*](https://arxiv.org/abs/2607.16051) and the tied-residual parameterization in [*On the Residual Scaling of Looped Transformers*](https://arxiv.org/abs/2606.18524).

Current conclusions:

- Residual scaling and 64-iteration training did not stabilize the unnormalized Sotaku loop.
- Two-block RMSNorm schedules are reliable and parameter-efficient, but peak around 85%.
- Randomized detached late states are the strongest reproducible training intervention. Delayed damping keeps three independent final checkpoints at 94.7-96.9% through iteration 4096.
- A second late supervised window is most useful after ordinary training has produced a healthy model. The first late-switch run reached 97.5% at 1024 without damping, but still degraded at deeper horizons.

## First-wave training matrix

Every arm uses the 20K SOTA testbed, seed 20260720, 16 supervised Sudoku iterations, the existing curriculum and averaged per-iteration cross-entropy loss, and the existing 128/1024-iteration probe. `baseline_unscaled` is the matched control. The residual arms keep the four stored blocks and the normal `ABCD` order but do not use outer RMSNorm. The schedule arms both store two blocks, apply exactly four blocks per Sudoku iteration, and use the recommended outer RMSNorm.

| arm | stored blocks | applications per Sudoku iteration | branch scale | prediction-feedback scale | outer state |
|---|---:|---|---:|---:|---|
| `baseline_unscaled` | 4 | `ABCD` | 1 | 1 | unbounded |
| `residual_linear_branches` | 4 | `ABCD` | 1/16 | 1 | unbounded |
| `residual_linear_all` | 4 | `ABCD` | 1/16 | 1/16 | unbounded |
| `residual_sqrt_all` | 4 | `ABCD` | 1/4 | 1/4 | unbounded |
| `schedule_abab` | 2 | `ABAB` | 1 | 1 | RMSNorm |
| `schedule_aabb` | 2 | `AABB` | 1 | 1 | RMSNorm |

The residual paper writes the multi-layer scale as `lambda / (N sqrt(m_L))`, where `m_L` is depth relative to the architecture used to tune the base learning rate. Sotaku keeps its reference depth fixed at four blocks, so `m_L = 1` and the paper-faithful loop factor is `1/N = 1/16`. The separate prediction-feedback addition has no counterpart in the paper, which is why the matrix tests both leaving it alone and scaling it with the other recurrent additions.

Run one job per detached invocation:

```bash
modal run --detach looping/modal_loop_ablation.py --arm residual-linear-all --trial 0
```

Results are written under `looping/` on the `sudoku-outputs` Modal volume. Replicate an arm with trials 1 and 2 only after the matched first wave shows that the arm learns normally and is competitive at the long-horizon probe.

The matched trial-0 results were:

| arm | final 16-iteration accuracy, 25K | final probe at 128 | final probe at 1024 | best 1024 probe |
|---|---:|---:|---:|---:|
| `baseline_unscaled` | 79.0% | 43.3% | 0.7% | 36.0% at 5K |
| `residual_linear_branches` | 77.6% | 5.2% | 0.0% | 0.5% at 3K |
| `residual_linear_all` | 76.9% | 76.3% | 3.1% | 79.8% at 9K |
| `residual_sqrt_all` | 78.0% | 6.3% | 0.0% | 65.9% at 7K |
| `schedule_abab` | 75.2% | 83.2% | 84.3% | 84.6% at 19K |
| `schedule_aabb` | 76.2% | 84.9% | 85.8% | 85.8% at 20K |

Residual scaling did not make the unnormalized Sotaku recurrence reliable. Scaling only the transformer branches badly unbalanced them against the unscaled prediction-feedback update. Scaling every recurrent addition by 1/16 allowed a healthy intermediate checkpoint, but the run still collapsed by 20K. The square-root control behaved similarly and collapsed earlier. This is direct evidence for Sotaku only; the residual-scaling paper studies a different looped-transformer parameterization without Sotaku's prediction-feedback path.

A true 64-training-iteration follow-up used branch and prediction-feedback scales of 1/64, with exact effective batch size 2048 via two 1024-example microbatches. It also failed: the best 1024 probe was 1.2% at step 2K and the final probe was 0.1%, while final 128-iteration accuracy was 76.0%. Simply training four times deeper with the paper-style tied-update scale therefore does not repair Sotaku's long recurrence.

Both compact RMSNorm schedules were healthy at the end. On full 25K-puzzle evaluation of each trial-0 best checkpoint, AABB retained a small advantage through 2048 iterations:

| schedule | parameters | 16 iterations | 128 iterations | 1024 iterations | 2048 iterations |
|---|---:|---:|---:|---:|---:|
| ABAB | 400,393 | 75.08% | 82.30% | 83.27% | 83.50% |
| AABB | 400,393 | 76.24% | 83.88% | 84.85% | 85.09% |

AABB led this full trial-0 comparison by 1.58 percentage points at 1024 iterations. Three paired training seeds give a more qualified result:

| schedule | best 1024 probe, trials 0/1/2 | mean best | mean final | mean best-to-final drop |
|---|---|---:|---:|---:|
| ABAB | 84.6% / 86.6% / 87.5% | 86.23% | 84.17% | 2.07 points |
| AABB | 85.8% / 85.1% / 86.6% | 85.83% | 85.60% | 0.23 points |

The schedules have indistinguishable peak accuracy at this sample size. AABB finishes more consistently and is the safer compact default, while ABAB produced the highest single probe. Both use about half the parameters of the four-block model and remain below the reliable four-block RMSNorm result.

## Detached late-state supervision

The first training intervention that directly improved the unnormalized model's long-horizon reliability was truncated training from randomized late states. On 80% of batches, training is unchanged: cross-entropy is averaged across iterations 1-16. On the other 20%, the model first runs without gradients to a randomly selected horizon in `{32, 64, 128, 256, 512}`. The hidden state and prediction feedback are detached, then the usual 16-iteration averaged cross-entropy is optimized from that state. This is replacement rather than an auxiliary loss: sampled late-state batches use only the late window, and gradients never pass through the burn-in.

Three independent 20K runs all finished healthy. Their final 1,000-puzzle probes scored 92.0%, 93.9%, and 96.4% at 1024 iterations; the selected probes were 93.3%, 93.9%, and 96.7%. The mean final score was 94.1%, so the result does not depend on harvesting a transient checkpoint. This is 3/3 successful final checkpoints, compared with 2/7 for the matched plain 20K recipe.

| selected checkpoint, undamped | 16 | 128 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| randomized late states, trial 0 | 76.32% | 92.34% | **93.55%** | 84.98% |
| randomized late states, trial 1 | 77.71% | 94.03% | **93.02%** | 69.98% |
| randomized late states, trial 2 | 77.89% | 94.26% | **96.68%** | 95.22% |
| start late states after step 2K | 77.31% | 93.49% | **95.08%** | 86.08% |

The first two selected checkpoints still deteriorated without an inference constraint, reaching 70-85% at 2048 and much lower scores at 4096. Trial 2 was naturally stronger at 95.22% at 2048, but its token RMS also kept growing. Randomized late states make useful long trajectories much more reproducible; they do not guarantee bounded dynamics by themselves.

Delayed damping complements the training intervention. Leaving iterations 1-512 unchanged and then using `alpha = 0.25` transferred across all three independent seeds without retuning. More importantly, it works on the final weights:

| final checkpoint with damping after 512 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| randomized late states, trial 0 | 93.61% | 94.92% | 94.85% | 94.86% | 94.82% |
| randomized late states, trial 1 | 93.83% | 94.80% | 94.74% | 94.71% | 94.69% |
| randomized late states, trial 2 | 94.36% | 96.76% | **96.84%** | **96.88%** | **96.91%** |

The three final checkpoints span 94.69-96.91% at 4096, with a mean of 95.47%. Their token RMS changes by at most 0.12 after damping begins. Accuracy stays flat or improves, so under-relaxation is stabilizing useful continued computation rather than merely freezing the iteration-512 answer. The selected trial-2 checkpoint reached 96.80% at 1024 and 96.88% at 4096; final weights were slightly better.

The ablations make the useful intervention fairly specific. A fixed 128 burn-in peaked at 82.9% and collapsed to 9.4% by the end. Recurrent RMSNorm plus randomized late states reached only 81.7%; compact AABB plus late states reached 79.4%; and increasing late-state batches from 20% to 50% finished healthy at 93.6% but was slower and did not improve the paired run. Randomized powers-of-two horizons matter more than simply applying more late losses or combining every stabilizer.

### Preliminary stay-solved screens

A 10K screening wave tested whether the model can explicitly learn to preserve a solved state. On a sampled late-state batch, `stay_recheck` trains the usual 16-step window, detaches its final state, advances without gradients by a random gap in `{16, 64, 256}`, then trains a second 16-step window. The two cross-entropy losses receive equal weight. `stay_consistency` adds a `0.1`-weight KL loss that asks the second window to preserve the first window's prediction distribution, but only on blank cells that the first window already answered correctly. Incorrect answers are never used as consistency targets.

The same wave separated two ways of introducing late states after step 2K. `ramp_after_2k` raises the late-batch probability from 0 to 20% over 4K steps. `curriculum_after_2k` uses 20% immediately but unlocks burn-in horizons 32, 64, 128, 256, and 512 at steps 2K through 6K. `clean_curriculum` combines the probability ramp and horizon unlock. All arms use seed 20260724.

| 10K screen | final 16, 25K | final 128, 1K | final 1024, 1K | best 1024 |
|---|---:|---:|---:|---:|
| ordinary randomized late states | 73.5% | 84.4% | 60.8% | 70.8% |
| probability ramp after 2K | 73.9% | 81.6% | 53.5% | 72.9% |
| horizon curriculum after 2K | **74.5%** | **84.1%** | 77.2% | 79.0% |
| ramp plus horizon curriculum | 73.9% | 83.4% | 72.4% | 74.9% |
| second late window | 72.8% | 81.9% | 78.6% | **79.5%** |
| second late window plus consistency | 72.8% | 79.8% | **79.4%** | 79.4% |

The second-window arms were also much steadier late in training. Across the five probes from steps 6K through the final checkpoint, `stay_recheck` averaged 77.8% at 1024 and never fell below 75.4%; `stay_consistency` averaged 77.3% and never fell below 74.9%. The matched control averaged 49.0% and fell as low as 0.8%. The consistency term did not raise the best 1024 score over recheck-only, but it reduced the final 128-to-1024 drop from 3.3 points to 0.4 points. At this sample size, the main useful intervention is the second supervised window; the extra consistency term is a possible refinement rather than a demonstrated requirement.

These are screening signals, not a new recommended recipe. The first wave compressed the warmup and every curriculum phase by half, which made the delayed step-2K start coincide with a training-data transition. The model had only 26.9-42.5% accuracy at 128 iterations at that handoff, so these runs do not cleanly answer what happens when late-state training begins from an already healthy model. `exp_stay_solved.py` now preserves the original 560-step warmup and full 0-4K hard-puzzle phase in `_healthy_screen` runs, and a regression test protects that ordering. A corrected screen and independent full-length replications are required before changing the default recipe.

### Budget calibration and 50K reliability cohort

The corrected 10K screen did not preserve the 20K ranking. With seed 20260724, ordinary randomized late states finished at 54.4% on the 1024-iteration probe while `stay_consistency` finished at 20.5%. At 20K the same arms finished at 89.4% and 94.8%, respectively, and `stay_consistency` peaked at 96.2%. The intervention that looked worst at 10K became best at 20K, so 10K screens are not accepted as rank-preserving proxies for these objectives.

The exact 50K schedule uses the historical 1,400-step warmup and 10K/10K/10K/20K curriculum phases. Trials 0-2 use seeds 20260724-20260726. Before launching trials 1-2, the cohort criteria were fixed as follows:

- A healthy final checkpoint scores at least 90% on the fixed 1,000-puzzle 1024-iteration probe.
- A run contains a harvestable checkpoint if any probe reaches at least 94%.
- Reliability comparisons report the mean and minimum of every 1024-iteration probe from step 30K through the final checkpoint, not only the peak.
- Any checkpoint used for a headline claim must also be evaluated on the full 25,000-puzzle set. The 1,000-puzzle probe is for monitoring and checkpoint selection.

Trial 0:

| 50K trial 0 | final 16, 25K | final 128 probe | final 1024 probe | best 1024 probe | 30K-final mean | 30K-final minimum |
|---|---:|---:|---:|---:|---:|---:|
| randomized late states | 80.78% | **97.0%** | **95.1%** | **97.7%** | **92.09%** | 41.9% |
| stay-consistency from step 0 | 80.45% | 96.2% | 91.1% | 95.5% | 90.87% | **73.4%** |

In trial 0, stay-consistency raised the late floor by 31.5 points but reduced the mean, peak, and final score. The best ordinary checkpoint's full-set profile was 95.65% / 97.07% / 88.42% / 31.59% at 128 / 1024 / 2048 / 4096; its final weights scored 96.51% / 94.85% / 56.01% / 6.28%.

Trials 1 and 2 completed the probe-level reliability cohort:

| recipe and trial | final 16, 25K | final 128 probe | final 1024 probe | best 1024 probe | 30K-final mean | 30K-final minimum |
|---|---:|---:|---:|---:|---:|---:|
| randomized late states, trial 1 | 80.56% | 95.3% | 97.2% | 98.7% | 96.73% | 92.1% |
| stay-consistency, trial 1 | 81.22% | 96.2% | 96.8% | 97.8% | 94.19% | 86.2% |
| randomized late states, trial 2 | 80.20% | 95.0% | 93.2% | 96.6% | 93.59% | 85.7% |
| stay-consistency, trial 2 | 80.24% | 95.7% | 97.5% | 97.7% | 95.64% | 87.2% |

All six 50K runs finished above the predefined 90% probe threshold and contained a checkpoint above 94%. Across three seeds, ordinary and stay-consistency training had nearly identical mean final probes, 95.17% and 95.13%. Starting consistency at step 0 reduced the worst observed late collapse, but did not improve the average or peak. Trials 1 and 2 have not received full 25K deep-horizon evaluation, so their numbers establish training reliability rather than headline accuracy.

The next experiment changed when the stay-solved objective begins. Both branches loaded the ordinary trial-0 checkpoint at step 39K, including optimizer and RNG state, and continued the same schedule through 50K. One branch kept ordinary randomized late-state training; the other enabled the second supervised window and consistency term only for the final 11K steps.

| continuation from the same 39K checkpoint | final 128 probe | final 1024 probe | best 1024 probe | branch mean | branch minimum |
|---|---:|---:|---:|---:|---:|
| ordinary continuation | 97.2% | 96.5% | 97.1% | 94.62% | 86.4% |
| switch on stay-consistency | **97.5%** | **97.5%** | **98.4%** | **97.36%** | **96.1%** |

The switched branch stayed between 96.1% and 98.4% at every probe from 40K through the final checkpoint. This is much stronger than applying stay-consistency from initialization: the ordinary objective first develops useful long-horizon computation, then the late objective teaches the model to preserve it. The ordinary branch did not replay the original continuation bit-for-bit despite restoring all saved RNG streams; small GPU-kernel numerical differences compound through training and recurrence, so full-set evaluation and independent replications remain necessary.

Full-set evaluation confirmed the 1024-iteration result and showed that the switched branch also has the deepest trajectory:

| full 25K evaluation | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| ordinary branch, best probe checkpoint | **96.38%** | 96.28% | 76.32% | 18.36% |
| switched branch, best probe checkpoint | 95.91% | 97.14% | **90.50%** | **35.22%** |
| switched branch, final checkpoint | 96.34% | **97.52%** | 88.98% | 30.20% |

The final switched weights are best at 1024; the step-43K checkpoint is better at 2048 and 4096. A 60-generation independent-sampling ES polish started from the step-43K checkpoint. Its 1,000-puzzle solved-and-settled probe moved from 95.0% to a best of 95.8% at generation 15, then ended at 94.5%. ES therefore did not improve this already-good seed. The recovered generation-15 weights and the checkpointing fix are documented in `es/EXPERIMENTS_ES.md`.

### Extending sampled states through 1024

Adding 1024 to the sampled burn-in set was substantially more expensive but produced the strongest trial-0 run. Its selected checkpoint scored 97.22% at 1024 undamped, then fell to 88.63% at 2048. The same fixed `alpha = 0.25` policy after iteration 512 converted that into 97.55%, 97.73%, and 97.81% at 1024, 2048, and 4096. The final weights were equally strong at 97.52%, 97.70%, and 97.88%, again removing checkpoint selection from the recipe.

The independent replication changed the conclusion. It used seed 20260721, matching standard late-state trial 1, and finished at 93.5% on its 128-iteration probe but only 88.7% at 1024; its best 1024 probe was 89.3%. The standard `{32, 64, 128, 256, 512}` run with the same seed finished at 93.9% at 1024. Adding 1024 to the training distribution therefore does not reliably improve long-horizon behavior.

The original damping policy also transferred poorly because the replicated model had already declined before iteration 512:

| final checkpoint with `alpha = 0.25` after 512 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| through-1024 trial 0 | 94.02% | 97.30% | **97.52%** | **97.70%** | **97.88%** |
| through-1024 trial 1 | 93.30% | 91.46% | 91.36% | 91.31% | 91.28% |

A paired policy sweep found a more conservative shared setting: leave iterations 1-128 unchanged, then use `alpha = 0.125`. Full 25,000-puzzle evaluation gave:

| final checkpoint with `alpha = 0.125` after 128 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| through-1024 trial 0 | 94.02% | 95.10% | 95.75% | 96.43% | **96.78%** |
| through-1024 trial 1 | 93.30% | 93.69% | 93.84% | 93.90% | **93.94%** |

Earlier damping reduces the two-seed spread and allows both models to keep improving, but the mean at 4096 is 95.36%, compared with 95.47% for the cheaper three-seed standard recipe. Its floor is also lower, 93.94% versus 94.69%. Sampling through 1024 is therefore a useful ceiling experiment, not the recommended default. On trial 0 alone, `alpha = 0.5` after 512 was a near tie with `alpha = 0.25`, and waiting until iteration 1024 to damp was worse; the replication shows why choosing a policy from that exceptional seed was misleading.

## Delayed recurrent damping

The strongest inference-time intervention leaves the first 128 recurrent iterations unchanged, then under-relaxes each later update with `alpha = 0.25`:

```text
proposed = F(hidden)
hidden = 0.75 * hidden + 0.25 * proposed
```

This is not an extra training loss and does not alter the checkpoint. On the fixed 1,000-puzzle confirmation set:

| checkpoint and policy | 128 iterations | 1024 iterations | 2048 iterations | 4096 iterations |
|---|---:|---:|---:|---:|
| naturally stable, undamped | 94.4% | 98.1% | 98.1% | 84.4% |
| naturally stable, damped after 128 | 94.4% | 95.8% | 96.2% | 96.4% |
| collapsed, undamped | 93.1% | 5.8% | 3.0% | 2.0% |
| collapsed, damped after 128 | 93.1% | 95.6% | 95.9% | 95.7% |

For the collapsed checkpoint, damping reduced mean token RMS at 1024 iterations from 189.8 to 101.4 and kept it at 102.7 through 4096. The rescue therefore is not merely equivalent to evaluating at an earlier iteration: accuracy continues to improve after iteration 128 while the state approaches a bounded trajectory.

Under-relaxation preserves a recurrence's fixed points while mapping each local Jacobian eigenvalue from `lambda` to `1 - alpha + alpha * lambda`. That mapping can stabilize oscillatory or overshooting modes. The stable checkpoint shows the tradeoff: damping costs 2.3 points at 1024, but prevents its much later 4096-iteration decline.

The full 25K-puzzle evaluation confirmed the effect and showed that it covers every difficulty bucket:

| checkpoint and policy | 16 iterations | 128 iterations | 1024 iterations | 2048 iterations |
|---|---:|---:|---:|---:|
| naturally stable, undamped | 81.83% | 95.31% | **98.85%** | **98.87%** |
| naturally stable, damped after 128 | 81.83% | 95.31% | 97.00% | 97.32% |
| collapsed, undamped | 80.62% | 93.39% | 5.64% | 3.06% |
| collapsed, damped after 128 | 80.62% | 93.39% | **95.66%** | **96.20%** |

On the collapsed checkpoint at 2048 iterations, per-bucket accuracy after damping was 99.84%, 99.54%, 92.80%, 92.84%, and 95.96% from easiest to hardest. The rescue is not an aggregate dominated by one subset. On the naturally stable checkpoint, damping is unnecessary and costs 1.55 points at 2048.

The nearby policy grid explains why the selected setting matters. Starting damping at 64 iterations sacrifices 128-iteration accuracy. Waiting until 256 preserves more of the original trajectory but allows instability to develop. With a 128-iteration warmup, `alpha = 0.5` falls to 55.7% by 2048, `0.375` to 78.3%, `0.25` reaches 95.9%, and `0.125` reaches 95.3%. For this checkpoint, 128 then 0.25 is the best tested balance.

The same policy was then applied across the clean-A training trajectory:

| checkpoint | undamped at 128 | undamped at 1024 | damped at 1024 | damped at 2048 |
|---:|---:|---:|---:|---:|
| 30K, healthy | 90.9% | 95.5% | 93.5% | 94.3% |
| 35K, collapsed | 90.4% | 2.8% | 84.0% | 80.2% |
| 40K, collapsed | 91.8% | 28.3% | 95.2% | 95.5% |
| 45K, collapsed | 93.5% | 7.9% | 96.2% | 96.6% |
| 50K, collapsed | 93.1% | 5.8% | 95.6% | 95.9% |

The 35K checkpoint needed stronger damping. With the same 128-iteration warmup and `alpha = 0.0625`, it scored 91.1%, 91.3%, and 91.4% at 1024, 2048, and 4096. Delayed damping therefore substantially rescued all four collapsed checkpoints from this run, but the useful alpha depends on the checkpoint.

Four checkpoints from the earlier ES rescue-boundary study provide the harder test. Selecting between `alpha = 0.25`, `0.125`, `0.0625`, and `0.03125` on the 1,000-puzzle probe gave:

| checkpoint | undamped at 128 | undamped at 1024 | selected damped at 1024 | selected damped at 4096 |
|---|---:|---:|---:|---:|
| cohort A 40K | 84.9% | 0.3% | 83.1% | 83.2% |
| cohort D 45K | 92.8% | 2.0% | 92.5% | 91.7% |
| cohort G 40K | 89.2% | 49.3% | 89.4% | 89.4% |
| clean-B 35K | 87.3% | 6.5% | 87.8% | 88.1% |

Small alpha preserves short-horizon behavior on these harder checkpoints, but usually does not improve on iteration 128. This is closer to slowing or freezing a good answer than uncovering useful longer computation. The practical inference rule is therefore:

1. Leave already stable checkpoints undamped.
2. For a collapse that remains healthy at 128, sweep `alpha` in `{0.25, 0.125, 0.0625, 0.03125}` after a fixed 128-iteration warmup on held-out puzzles.
3. Check both 1024 and 2048. Prefer a setting only when accuracy improves beyond the 128-iteration result; otherwise, evaluating directly at 128 is simpler and faster.

## Sudoku Jacobian and loss-gradient diagnostics

`eval_loop_diagnostics.py` adapts [Anthropic's Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) to Sudoku. It rolls a checkpoint to each requested horizon without gradients, differentiates only the next recurrent Sudoku iteration, and averages the causal direction for each of the nine digit logits at every block boundary. It separately differentiates the correct-answer margin, preserving puzzle and cell identity, and computes the 16-by-16 cosine matrix between the parameter gradients from the 16 supervised cross-entropy losses.

The 100-puzzle reference comparison gave:

| checkpoint | solved at 16 | solved at 128 | solved at 1024 | answer-margin gradient cosine, 1024 vs 16 | digit-lens CKA, 1024 vs 16 |
|---|---:|---:|---:|---:|---:|
| naturally stable, unbounded | 78/100 | 95/100 | 100/100 | 0.466 | 0.996 |
| collapsed, unbounded | 82/100 | 91/100 | 8/100 | 0.298 | 0.970 |
| reliable RMSNorm | 83/100 | 91/100 | 92/100 | 0.680 | 0.997 |

The broad nine-digit causal subspace therefore remains nearly unchanged even when answers collapse. The puzzle-specific answer-margin direction is more informative: RMSNorm preserves it best, while the collapsed model rotates much farther away from its iteration-16 direction.

A same-run comparison removes initialization as a confound. The clean-A run was healthy at step 30K and collapsed by 35K:

| clean-A checkpoint | solved at 128 | solved at 1024 | answer-margin gradient cosine, 1024 vs 16 | digit-lens CKA, 1024 vs 16 |
|---:|---:|---:|---:|---:|
| 30K | 89/100 | 95/100 | 0.498 | 0.993 |
| 35K | 89/100 | 4/100 | 0.234 | 0.994 |
| 40K | 90/100 | 18/100 | 0.374 | 0.996 |
| 45K | 95/100 | 6/100 | 0.319 | 0.982 |
| 50K | 91/100 | 8/100 | 0.298 | 0.970 |

The late-state comparison points in the same direction:

| final checkpoint | solved at 16 | solved at 128 | solved at 1024 | answer-margin gradient cosine, 1024 vs 16 | digit-lens CKA, 1024 vs 16 | state RMS at 1024 |
|---|---:|---:|---:|---:|---:|---:|
| randomized states through 512, trial 2 | 76/100 | 92/100 | 96/100 | 0.356 | 0.996 | 142.9 |
| randomized states through 1024, trial 0 | 79/100 | 93/100 | 97/100 | 0.470 | 0.995 | 205.4 |

Extending sampled states to 1024 improves preservation of the puzzle-specific answer direction while leaving the broad digit subspace unchanged. Its hidden states are substantially larger, so lower magnitude is not the cause of the accuracy gain; delayed damping bounds that already-useful computation afterward.

The iteration-loss gradient matrix did not show a corresponding reduction in conflict. For the clean-A transformer-block parameters, the mean early-versus-late cosine changed from 0.080 at 30K to 0.033 at 35K, while the fraction of negative off-diagonal pairs decreased from 0.200 to 0.142. In the two late-state models above, extending through 1024 changed the early-versus-late cosine from 0.051 to -0.058 and increased the negative-pair fraction from 0.117 to 0.350 despite improving long-horizon accuracy. Gradient conflict may still contribute to optimization noise, but these diagnostics do not support it as the immediate cause of collapse. The sharper description is that training changes a narrow, puzzle-specific causal direction while leaving the general digit-output geometry intact.
