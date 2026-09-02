# Looped Model Experiments

This folder contains controlled follow-ups to two July 2026 results: the layer-loop schedule in [*Loop the Loopies!*](https://arxiv.org/abs/2607.16051) and the tied-residual parameterization in [*On the Residual Scaling of Looped Transformers*](https://arxiv.org/abs/2606.18524).

Current conclusions:

- Residual scaling and 64-iteration training did not stabilize the unnormalized Sotaku loop.
- Two-block RMSNorm schedules are reliable and parameter-efficient, but peak around 85%.
- Training on randomly sampled later iterations is the recommended default. It changes neither the architecture nor the loss, and the standard 20K and 50K runs tested here retained high final accuracy at 1024. The recommended 50K final checkpoint scores 99.12% at 1024 and 98.63% at 4096 in FP32, without damping. Its historical BF16 scores were 98.80% at 1024 and 92.98% at 2048.
- Adding a second supervised window helped most when introduced after the model had already reached high accuracy. Starting the extra objective from initialization did not improve the mean result.
- Recovery tests and interventions on state updates locate the observed failure where an incorrect digit's score overtakes the correct digit's score. State magnitude, total hidden-state rotation, and gradient conflict do not reliably distinguish models that retain correct predictions.
- The highest-scoring July BF16 research checkpoint combines training on later iterations through step 39K with a second future cross-entropy window and calibrated minimum-margin loss through step 50K. Its final weights reached 99.00% at 1024, 98.46% at 2048, and 82.17% at 4096 without inference adjustments. This staged result has not yet been replicated; later-iteration training alone remains the recommended public recipe.
- Delayed damping remains an optional inference setting for checkpoints whose accuracy declines at larger iteration counts. It keeps three independently trained final checkpoints at 94.69-96.91% at iteration 4096, but is unnecessary when ordinary inference already meets the desired accuracy at the target iteration count.
- Use 20K runs for routine comparisons, then validate promising changes at 50K. Training on later iterations improved the 20K results across three random seeds, while 10K did not reliably preserve later rankings. A 20K result can still miss a positive or negative phase change after step 20K.

All current runs use the first 2.7M puzzles from the training split. The full schedule has 50K optimizer steps, batch size 2048, a 1,400-step warmup, and actual rating pools 51+, 11+, 1+, and 0+ over phases of 10K, 10K, 10K, and 20K steps. Historical config and log labels say 21+ and 6+ for the first two phases, but the trainer selects whole buckets by their lower endpoint, producing 51+ and 11+. The training behavior is unchanged. The 20K schedule is useful for preliminary comparisons; 10K did not preserve the later ranking, and the current recipe has not been tested with a smaller puzzle pool while holding other settings fixed.

An iteration is one pass through the shared model blocks; a training step is an optimizer update. Monitoring evaluations use 1,000 fixed test puzzles and are called probes in historical logs. Full evaluations use 25,000 puzzles. A score can fall across training checkpoints or across inference iterations; the tables specify which changes. Experiment IDs and filenames retain their historical names.

## September Release Checks

The release audit reproduced the recommended later-iteration training checkpoint's full BF16 25K counts exactly at 128/1024/2048/4096. The same weights in FP32 score 96.288 / 99.116 / 99.048 / 98.632%. Compiled BF16 scores 96.372 / 98.952 / 97.600 / 92.512%. The historical profiles below are therefore execution-specific: do not attribute their deep-iteration deterioration entirely to learned dynamics. The [precision study](../release/PRECISION_RESULTS.md) holds weights and puzzles fixed; FP32 is now the public inference default, with no retraining or damping.

A separate [burn-in dropout experiment](BURNIN_DROPOUT.md) completed four matched 4K continuations from two high-accuracy starting checkpoints, retaining the original 50K schedule and supervised-window dropout. Turning off dropout only during the initial gradient-free iterations lowered both runs' minimum BF16 monitoring scores and both full-set 2048 results. FP32 reduced the difference but did not make dropout-off consistently better. Keep dropout on; no 50K extensions were justified. Release engineering fixes and fresh reference results are in [the audit](../V2_RELEASE_AUDIT.md).

## Initial Training Comparisons

Every variant uses the original 20K training setup, random seed 20260720, 16 supervised Sudoku iterations, the existing curriculum and averaged per-iteration cross-entropy, and monitoring at 128/1024 iterations. `baseline_unscaled` is the original model without added normalization or scaling. The residual-scaling variants keep its four stored blocks and normal `ABCD` order. The layer-order variants store two blocks, apply exactly four blocks per Sudoku iteration, and use RMSNorm after each complete iteration.

| variant | stored blocks | applications per Sudoku iteration | branch scale | prediction-feedback scale | state normalization |
|---|---:|---|---:|---:|---|
| `baseline_unscaled` | 4 | `ABCD` | 1 | 1 | none |
| `residual_linear_branches` | 4 | `ABCD` | 1/16 | 1 | none |
| `residual_linear_all` | 4 | `ABCD` | 1/16 | 1/16 | none |
| `residual_sqrt_all` | 4 | `ABCD` | 1/4 | 1/4 | none |
| `schedule_abab` | 2 | `ABAB` | 1 | 1 | RMSNorm |
| `schedule_aabb` | 2 | `AABB` | 1 | 1 | RMSNorm |

The residual paper writes the multi-layer scale as `lambda / (N sqrt(m_L))`, where `m_L` is depth relative to the architecture used to tune the base learning rate. Sotaku keeps its reference depth fixed at four blocks, so `m_L = 1` and the paper's loop factor is `1/N = 1/16`. The separate prediction-feedback addition has no counterpart in the paper, so the comparisons test both leaving it alone and scaling it with the other recurrent additions.

Run one job per detached invocation:

```bash
modal run --detach looping/modal_loop_ablation.py --arm residual-linear-all --trial 0
```

Results are written under `looping/` on the `sudoku-outputs` Modal volume. Replicate a variant with trials 1 and 2 only after its initial matched comparison shows competitive 1024-iteration accuracy.

The matched trial-0 results were:

| arm | final 16-iteration accuracy, 25K | final probe at 128 | final probe at 1024 | best 1024 probe |
|---|---:|---:|---:|---:|
| `baseline_unscaled` | 79.0% | 43.3% | 0.7% | 36.0% at 5K |
| `residual_linear_branches` | 77.6% | 5.2% | 0.0% | 0.5% at 3K |
| `residual_linear_all` | 76.9% | 76.3% | 3.1% | 79.8% at 9K |
| `residual_sqrt_all` | 78.0% | 6.3% | 0.0% | 65.9% at 7K |
| `schedule_abab` | 75.2% | 83.2% | 84.3% | 84.6% at 19K |
| `schedule_aabb` | 76.2% | 84.9% | 85.8% | 85.8% at 20K |

Residual scaling did not make the unnormalized Sotaku recurrence reliable. Scaling only the transformer branches unbalanced them against the unscaled prediction-feedback update. Scaling every recurrent addition by 1/16 reached an intermediate 79.8% at 1024, but fell to 3.1% by training step 20K. The square-root scaling variant lost accuracy earlier. This is evidence for Sotaku only; the residual-scaling paper studies a different looped-transformer parameterization without Sotaku's prediction-feedback path.

A true 64-training-iteration follow-up used branch and prediction-feedback scales of 1/64, with exact effective batch size 2048 via two 1024-example microbatches. It also failed: the best 1024 probe was 1.2% at step 2K and the final probe was 0.1%, while final 128-iteration accuracy was 76.0%. Simply training four times deeper with the paper-style tied-update scale therefore does not repair Sotaku's long recurrence.

Both compact RMSNorm schedules finished above 84% on the 1024-iteration monitoring sample. On full 25K-puzzle evaluation of each trial-0 best checkpoint, AABB retained a small advantage through 2048 iterations:

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

## Training On Later Iterations

On 80% of batches, training is unchanged: cross-entropy is averaged across iterations 1-16. On the other 20%, the model first runs without gradients for a randomly selected number of iterations in `{32, 64, 128, 256, 512}`. It then trains through the next 16 iterations using the same averaged cross-entropy. For example, a 128-iteration initial segment is followed by loss on iterations 129-144. The hidden state and prediction feedback are detached at that boundary, meaning gradients do not flow into the initial segment, also called burn-in. This replaces the ordinary window on selected batches; it adds no auxiliary loss. Historical configurations call the method `late_state_ce`.

Three independent 20K runs finished at 92.0%, 93.9%, and 96.4% on their 1,000-puzzle 1024-iteration monitoring samples; the selected checkpoints scored 93.3%, 93.9%, and 96.7%. The mean final score was 94.1%, so the improvement does not depend on selecting a briefly accurate intermediate checkpoint. All 3 final checkpoints exceeded 90%. The original 20K recipe passed the earlier stabilization study's lower 80% threshold in only 2 of 7 runs.

| selected checkpoint, undamped | 16 | 128 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| later-iteration training, trial 0 | 76.32% | 92.34% | **93.55%** | 84.98% |
| later-iteration training, trial 1 | 77.71% | 94.03% | **93.02%** | 69.98% |
| later-iteration training, trial 2 | 77.89% | 94.26% | **96.68%** | 95.22% |
| start late states after step 2K | 77.31% | 93.49% | **95.08%** | 86.08% |

Without damping, the first two selected checkpoints still fell to 70-85% at 2048 and much lower scores at 4096. Trial 2 was stronger at 95.22% at 2048, but its per-cell state RMS also kept growing. Training on randomly sampled later iterations improved reproducibility; it did not guarantee bounded state magnitude or indefinitely correct predictions.

Delayed damping complements the training intervention. Leaving iterations 1-512 unchanged and then using `alpha = 0.25` transferred across all three independent seeds without retuning. More importantly, it works on the final weights:

| final checkpoint with damping after 512 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| later-iteration training, trial 0 | 93.61% | 94.92% | 94.85% | 94.86% | 94.82% |
| later-iteration training, trial 1 | 93.83% | 94.80% | 94.74% | 94.71% | 94.69% |
| later-iteration training, trial 2 | 94.36% | 96.76% | **96.84%** | **96.88%** | **96.91%** |

The three final checkpoints span 94.69-96.91% at 4096, with a mean of 95.47%. Their token RMS changes by at most 0.12 after damping begins. Accuracy stays flat or improves, so under-relaxation is stabilizing useful continued computation rather than merely freezing the iteration-512 answer. The selected trial-2 checkpoint reached 96.80% at 1024 and 96.88% at 4096; final weights were slightly better.

The comparisons make the useful intervention fairly specific. A fixed 128-iteration burn-in peaked at 82.9% and fell to 9.4% by the end. Recurrent RMSNorm plus training on random later iterations reached only 81.7%; compact AABB plus that training reached 79.4%. Increasing the later-iteration batches from 20% to 50% finished at 93.6% but was slower and did not improve on the paired run. In these tests, sampling the initial iteration count from the powers of two worked better than a fixed count or combining the training with normalization.

### Training To Preserve Correct Predictions

A 10K preliminary comparison tested whether extra training later in the same trajectory would help preserve correct answers. The second-window variant, `stay_recheck`, trains the usual 16-iteration window on a sampled later-iteration batch, detaches its final state, advances without gradients by a random gap in `{16, 64, 256}`, then trains a second 16-iteration window. The two cross-entropy losses receive equal weight. The second-window-plus-consistency variant, `stay_consistency`, also adds a KL-divergence loss with weight `0.1`. This penalty asks later predictions to match the first window's final probability distribution across all nine digits, only on originally blank cells answered correctly at that first endpoint. It is not just a penalty for changing the chosen digit. Incorrect first-window answers are never consistency targets.

The same comparison separated two ways of introducing later-iteration training after step 2K. `ramp_after_2k` raises its batch probability from 0 to 20% over 4K steps. `curriculum_after_2k` uses 20% immediately but progressively allows initial gradient-free segments of 32, 64, 128, 256, and 512 iterations at steps 2K through 6K. `clean_curriculum` combines the probability ramp and increasing iteration counts. All variants use random seed 20260724.

| 10K preliminary run | final 16, 25K | final 128, 1K | final 1024, 1K | best 1024 |
|---|---:|---:|---:|---:|
| later-iteration training alone | 73.5% | 84.4% | 60.8% | 70.8% |
| probability ramp after 2K | 73.9% | 81.6% | 53.5% | 72.9% |
| increase initial iteration counts after 2K | **74.5%** | **84.1%** | 77.2% | 79.0% |
| probability ramp plus increasing iteration counts | 73.9% | 83.4% | 72.4% | 74.9% |
| second late window | 72.8% | 81.9% | 78.6% | **79.5%** |
| second late window plus consistency | 72.8% | 79.8% | **79.4%** | 79.4% |

The second-window arms were also much steadier late in training. Across the five probes from steps 6K through the final checkpoint, `stay_recheck` averaged 77.8% at 1024 and never fell below 75.4%; `stay_consistency` averaged 77.3% and never fell below 74.9%. The matched control averaged 49.0% and fell as low as 0.8%. The consistency term did not raise the best 1024 score over recheck-only, but it reduced the final 128-to-1024 drop from 3.3 points to 0.4 points. At this sample size, the main useful intervention is the second supervised window; the extra consistency term is a possible refinement rather than a demonstrated requirement.

These preliminary results did not justify a new recommended recipe. The first comparison compressed the warmup and every curriculum phase by half, which made the delayed step-2K start coincide with a training-data transition. The model had only 26.9-42.5% accuracy at 128 iterations at that point, so these runs do not answer what happens when training on later iterations begins from an already accurate model. `exp_stay_solved.py` now preserves the original 560-step warmup and full 0-4K hard-puzzle phase in `_healthy_screen` runs, and a regression test protects that ordering. The corrected comparison and longer runs follow below.

### Training Duration And Repeated 50K Runs

The corrected 10K comparison did not preserve the 20K ranking. With random seed 20260724, later-iteration training alone finished at 54.4% on the 1024-iteration monitoring sample while `stay_consistency` finished at 20.5%. At 20K the same variants finished at 89.4% and 94.8%, respectively, and `stay_consistency` peaked at 96.2%. The intervention that looked worst at 10K became best at 20K, so a 10K comparison cannot reliably predict which of these objectives will work best in longer training.

The exact 50K schedule uses the historical 1,400-step warmup and 10K/10K/10K/20K curriculum phases. Trials 0-2 use random seeds 20260724-20260726. Before launching trials 1-2, the comparison criteria were fixed as follows:

- A final checkpoint passes if it scores at least 90% on the fixed 1,000-puzzle evaluation at iteration 1024.
- Separately, record whether any saved checkpoint reaches at least 94% on that evaluation.
- Report the mean and minimum of every 1024-iteration monitoring score from training step 30K through the final checkpoint, not only the peak.
- Any checkpoint used for a headline claim must also be evaluated on the full 25,000-puzzle set. The 1,000-puzzle evaluation is for monitoring and checkpoint selection.

Trial 0:

| 50K trial 0 | final 16, 25K | final 128 probe | final 1024 probe | best 1024 probe | 30K-final mean | 30K-final minimum |
|---|---:|---:|---:|---:|---:|---:|
| later-iteration training | 80.78% | **97.0%** | **95.1%** | **97.7%** | **92.09%** | 41.9% |
| second window plus consistency from step 0 | 80.45% | 96.2% | 91.1% | 95.5% | 90.87% | **73.4%** |

In trial 0, the second window plus consistency raised the minimum observed score from step 30K onward by 31.5 points but reduced the mean, peak, and final score. The best checkpoint trained without those additions had a full-set profile of 95.65% / 97.07% / 88.42% / 31.59% at 128 / 1024 / 2048 / 4096; its final weights scored 96.51% / 94.85% / 56.01% / 6.28%.

Trials 1 and 2 completed the repeated-run comparison on the 1K monitoring samples:

| recipe and trial | final 16, 25K | final 128 probe | final 1024 probe | best 1024 probe | 30K-final mean | 30K-final minimum |
|---|---:|---:|---:|---:|---:|---:|
| later-iteration training, trial 1 | 80.56% | 95.3% | 97.2% | 98.7% | 96.73% | 92.1% |
| second window plus consistency, trial 1 | 81.22% | 96.2% | 96.8% | 97.8% | 94.19% | 86.2% |
| later-iteration training, trial 2 | 80.20% | 95.0% | 93.2% | 96.6% | 93.59% | 85.7% |
| second window plus consistency, trial 2 | 80.24% | 95.7% | 97.5% | 97.7% | 95.64% | 87.2% |

All six 50K runs finished above the predefined 90% monitoring threshold and contained a checkpoint above 94%. Across three random seeds, later-iteration training alone and the added second window plus consistency had nearly identical mean final monitoring scores, 95.17% and 95.13%. Starting the additions at step 0 reduced the worst observed late-training drop, but did not improve the average or peak. Trials 1 and 2 have not received full 25K evaluation at large iteration counts, so their numbers describe reliability on the monitoring samples, not headline benchmark accuracy.

The next experiment delayed the additions. Both branches loaded the trial-0 checkpoint trained on later iterations at step 39K, including optimizer and RNG state, and continued the same schedule through 50K. One branch kept the objective unchanged; the other enabled the second supervised window and consistency term only for the final 11K training steps.

| continuation from the same 39K checkpoint | final 128 probe | final 1024 probe | best 1024 probe | branch mean | branch minimum |
|---|---:|---:|---:|---:|---:|
| ordinary continuation | 97.2% | 96.5% | 97.1% | 94.62% | 86.4% |
| add second window and consistency | **97.5%** | **97.5%** | **98.4%** | **97.36%** | **96.1%** |

The switched branch stayed between 96.1% and 98.4% at every monitoring evaluation from 40K through the final checkpoint. Delaying the additions worked better in this run than enabling them from initialization. The unchanged-objective branch did not replay the original continuation bit-for-bit despite restoring all saved RNG streams; small GPU-kernel numerical differences compound through training and recurrence, so full-set evaluation and independent replications remain necessary.

Full-set evaluation confirmed the 1024-iteration improvement. The switched branch also retained more accuracy at 2048 and 4096:

| full 25K evaluation | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| ordinary branch, best probe checkpoint | **96.38%** | 96.28% | 76.32% | 18.36% |
| switched branch, best probe checkpoint | 95.91% | 97.14% | **90.50%** | **35.22%** |
| switched branch, final checkpoint | 96.34% | **97.52%** | 88.98% | 30.20% |

The final switched weights are best at 1024; the step-43K checkpoint is better at 2048 and 4096. A 60-generation independent-sampling ES fine-tuning run started from the step-43K checkpoint. ES rewarded correct blank-cell predictions at iteration 2048 that matched those at 1920, without checking intervening predictions. On the 1,000-puzzle monitoring sample, that score moved from 95.0% to a best of 95.8% at generation 15, then ended at 94.5%. The final model was therefore worse than the starting checkpoint by this measure. The recovered generation-15 weights and the checkpointing fix are documented in [the ES notes](../es/EXPERIMENTS_ES.md).

### Extending sampled states through 1024

Adding 1024 to the sampled burn-in set was substantially more expensive but produced the strongest trial-0 run. Its selected checkpoint scored 97.22% at 1024 undamped, then fell to 88.63% at 2048. The same fixed `alpha = 0.25` policy after iteration 512 converted that into 97.55%, 97.73%, and 97.81% at 1024, 2048, and 4096. The final weights were equally strong at 97.52%, 97.70%, and 97.88%, again removing checkpoint selection from the recipe.

The independent replication changed the conclusion. It used seed 20260721, matching standard later-iteration training trial 1, and finished at 93.5% on its 128-iteration probe but only 88.7% at 1024; its best 1024 probe was 89.3%. The standard `{32, 64, 128, 256, 512}` run with the same seed finished at 93.9% at 1024. Adding 1024 to the training distribution therefore does not reliably improve long-horizon behavior.

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

Earlier damping reduces the spread between the two random seeds and allows both models to keep improving, but their mean at 4096 is 95.36%, compared with 95.47% for the cheaper three-seed standard recipe. Their lowest score is also lower, 93.94% versus 94.69%. Sampling through 1024 can achieve a high score in one run, but is not the recommended default. On trial 0 alone, `alpha = 0.5` after 512 was a near tie with `alpha = 0.25`, and waiting until iteration 1024 to damp was worse; the replication shows why choosing a setting from that exceptional run was misleading.

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

For the checkpoint whose accuracy collapsed, damping reduced mean per-cell state RMS at 1024 iterations from 189.8 to 101.4 and kept it at 102.7 through 4096. The improvement is not merely equivalent to evaluating at an earlier iteration: accuracy continues to improve after iteration 128 while state magnitude changes little over the measured later iterations.

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

1. Leave checkpoints undamped when ordinary inference already meets the desired accuracy at the target iteration count.
2. For a checkpoint that is accurate at 128 but deteriorates later, sweep `alpha` in `{0.25, 0.125, 0.0625, 0.03125}` after a fixed 128-iteration warmup on held-out puzzles.
3. Check both 1024 and 2048. Prefer a setting only when accuracy improves beyond the 128-iteration result; otherwise, evaluating directly at 128 is simpler and faster.

## Sudoku Jacobian and loss-gradient diagnostics

`eval_loop_diagnostics.py` adapts [Anthropic's Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) to Sudoku. It runs a checkpoint to each requested iteration without gradients, differentiates only the next recurrent Sudoku iteration, and averages the gradient direction for each of the nine digit logits at every block boundary. It separately differentiates the correct-answer margin, preserving puzzle and cell identity, and computes the 16-by-16 cosine matrix between the parameter gradients from the 16 supervised cross-entropy losses. A gradient direction describes which small changes to the current state would most change the measured output; it does not describe the entire future trajectory.

The 100-puzzle reference comparison gave:

| checkpoint | solved at 16 | solved at 128 | solved at 1024 | answer-margin gradient cosine, 1024 vs 16 | digit-lens CKA, 1024 vs 16 |
|---|---:|---:|---:|---:|---:|
| accurate original checkpoint, no state normalization | 78/100 | 95/100 | 100/100 | 0.466 | 0.996 |
| failing original checkpoint, no state normalization | 82/100 | 91/100 | 8/100 | 0.298 | 0.970 |
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

The comparison of models trained on later iterations points in the same direction:

| final checkpoint | solved at 16 | solved at 128 | solved at 1024 | answer-margin gradient cosine, 1024 vs 16 | digit-lens CKA, 1024 vs 16 | state RMS at 1024 |
|---|---:|---:|---:|---:|---:|---:|
| randomized states through 512, trial 2 | 76/100 | 92/100 | 96/100 | 0.356 | 0.996 | 142.9 |
| randomized states through 1024, trial 0 | 79/100 | 93/100 | 97/100 | 0.470 | 0.995 | 205.4 |

Extending sampled states to 1024 improves preservation of the puzzle-specific answer direction while leaving the broad digit subspace unchanged. Its hidden states are substantially larger, so lower magnitude is not the cause of the accuracy gain; delayed damping bounds that already-useful computation afterward.

The iteration-loss gradient matrix did not show a corresponding reduction in conflict. For the clean-A transformer-block parameters, the mean early-versus-late cosine changed from 0.080 at 30K to 0.033 at 35K, while the fraction of negative off-diagonal pairs decreased from 0.200 to 0.142. In the two models trained on later iterations above, extending through 1024 changed the early-versus-late cosine from 0.051 to -0.058 and increased the negative-pair fraction from 0.117 to 0.350 despite improving long-horizon accuracy. Gradient conflict may still contribute to optimization noise, but these diagnostics do not support it as the immediate cause of collapse. The sharper description is that training changes a narrow, puzzle-specific causal direction while leaving the general digit-output geometry intact.

## Recovery and update-component diagnostics

`eval_recovery_diagnostics.py` follows the same 1,000 balanced test puzzles for 16 additional iterations from several horizons. It records solved-puzzle retention, unsolved recovery, prediction changes, hidden-state direction, and the minimum correct-answer margin. At iteration 512, the naturally stable, RMSNorm, and later-iteration training checkpoints retained every solved puzzle through iteration 528. The collapsed clean-A checkpoint retained only 89.9%; 25.8% of all puzzles became worse while 5.4% improved. Its 10th-percentile directional minimum margin was already negative and continued falling.

Delaying the switch to training on later iterations shows what the objective does and does not learn. The step-12K checkpoint scored 62.0% at iteration 1024 and fell to 59.5% over the next 16 iterations. Some puzzles with 1-2 or 3-10 incorrect blank cells at the start of that window improved; historical logs call these groups `near_miss` and `semi_bad`. However, only 92.7% of initially solved puzzles were still solved at the end. At step 13K, iteration-1024 accuracy recovered to 94.5% and that retention rate rose to 99.8%; recovery of puzzles with 3-10 errors was still negligible. The later-iteration objective mainly improved preservation of correct answers, not recovery from arbitrary incorrect states.

`eval_update_decomposition.py` splits each complete recurrent update into a radial component, which changes hidden-state size, and a tangential component, which changes direction. After an untouched 128-iteration warmup, it scales one component at a time:

| clean-A checkpoint | undamped 1024 | radial only at 0.25 | tangential only at 0.25 |
|---|---:|---:|---:|
| 30K, healthy | 96.6% | 95.7% | 94.3% |
| 35K, collapsed | 2.6% | 11.1% | 77.2% |
| 40K, collapsed | 25.1% | 5.1% | 94.7% |
| 45K, collapsed | 6.2% | 1.5% | 96.2% |
| 50K, collapsed | 5.1% | 0.8% | 95.3% |

Reducing radial growth does not repair the accuracy drop and often makes it worse. Reducing the complete direction-changing update raises three failing checkpoints to 94.7-96.2% and substantially improves the harder 35K checkpoint. Changing that component therefore affects this failure, but the amount of rotation alone does not predict accuracy. RMSNorm retains correct answers with much larger per-iteration direction changes, and models trained on later iterations can match a failing checkpoint's angular rate. Accurate models differ in how much they turn; the common requirement is that every blank cell's correct digit still scores above its strongest incorrect competitor.

`eval_update_source_decomposition.py` separately scales direction changes from prediction feedback and from transformer layers. Scaling either source alone breaks their balance and can collapse the naturally stable model. Scaling both sources is better but still weaker than scaling the complete recurrent update. The evidence does not support blaming one recurrent subcomponent.

The observed failure is specific: many small later updates accumulate until an incorrect digit overtakes the correct digit in one or more cells. Delayed damping slows that change, while RMSNorm and training on later iterations produce different trajectories that retain positive margin. These observations motivated training losses that protect future minimum answer margin or preserve correct predictions. They did not establish a universal root cause or a need to force a fixed hidden state, a particular norm, or a uniformly small rotation.

### Trajectory dimension and smoothness

`eval_trajectory_geometry.py` sampled 50 balanced test puzzles and compared a naturally stable plain checkpoint, a collapsed plain checkpoint, the standalone later-iteration training checkpoint, and the combined margin checkpoint. It measured recurrent updates over 16-step windows beginning at iterations 16, 128, 512, and 1024.

Late updates are temporally smooth in every model, including the failing checkpoint: consecutive-update cosine at iteration 1024 was 0.990-1.000. Smooth motion therefore does not guarantee correct answers. The relative change in the update was nevertheless much larger for the failing model: 0.0430 at iteration 1024, compared with 0.0066 for the accurate original checkpoint, 0.0039 for later-iteration training, and 0.0016 for the combined checkpoint.

The compact structure is shared most clearly in feature space, not as one whole-board path. A 16-component basis fitted to token updates from half the puzzles explained 87.6-97.3% of token-update variance on held-out puzzles at iteration 1024. By contrast, a 64-component basis fitted to whole-board updates explained only 8.5-19.1% on held-out puzzles. An early token basis also retained 75-97% explanatory power at iteration 1024. The evidence is consistent with a small shared set of feature-space motions whose strengths and cell locations depend on the puzzle. It does not establish one universal low-dimensional board trajectory, and the collapsed checkpoint shows that low dimension and straight motion alone do not prevent wrong answers.

### Held-out recurrent-state geometry study

The [twelve-part geometry study](trajectory_viz/study/README.md) followed this diagnostic with 20 discovery, 20 validation, and 20 final puzzles per analysis, balanced across rating buckets. Each arm used frozen analysis choices, shuffled controls, random projections, and checkpoint transfer where applicable. It compared the stable plain, collapsed plain, standalone later-iteration training, and combined margin checkpoints.

The state contains substantial current Sudoku information. Answer margin is decoded within each checkpoint with held-out R-squared values of 0.947-0.987. Current expected conflicts are decoded with R-squared values of 0.860-0.945, and candidate-set size remains readable after subtracting direct input-symbol effects. A board-level solve-progress coordinate also transfers to unseen puzzles, with mean within-puzzle Spearman correlation 0.811. Digit identity occupies a compact categorical subspace, but natural numeric and cyclic digit orders do not beat shuffled orders.

The global-shape hypotheses did not survive the controls. The study found no shared helix, loop, arc, oscillatory mode, hidden-state fixed point, direction that reliably improves solve progress when used to modify the state, or transferable puzzle-level collapse warning. Projection selection alone raised a synthetic null's apparent helix score from 0.16 to 0.52. Margin, conflict, and digit axes were useful within a checkpoint but generally failed direct checkpoint transfer or were matched by random low-rank subspaces.

The clearest distinction in this comparison is how much the state direction changes at later iterations. Raw update norms remain large through iteration 1024, so even the accurate models are not approaching a fixed hidden state over the measured interval. From iterations 896 to 1024, normalized-state movement was 0.0064 for the accurate original checkpoint, 0.0254 for later-iteration training, and 0.0189 for the combined model, versus 0.1027 for the failing model. Relative acceleration showed the same separation. All four trajectories are smooth and nearly straight; the failing trajectory keeps changing direction enough for incorrect digits to overtake correct ones.

Useful follow-up measurements are weakest correct-answer margin, normalized-state movement, relative acceleration, the fraction of solved puzzles that remain solved, and prediction changes. Apparent PCA shape, raw update magnitude, smoothness alone, and proximity to a hidden-state fixed point did not reliably distinguish accurate models in this study. Because all final puzzle-level failures in the early-warning analysis came from one checkpoint, independent failing training runs are still required before claiming a general early-warning rule.

<a id="staged-recheck-and-margin"></a>

### Additional Training Window And Margin Penalty

These experiments remain research alternatives, not part of the recommended later-iteration training recipe.

`eval_margin_floor_calibration.py` measured the initial gradient-free iterations, first 16-iteration training window, gradient-free gap, and second 16-iteration window used by the earlier prediction-preservation experiment. The margin is the correct digit's logit minus the largest incorrect-digit logit. On the high-accuracy step-39K checkpoint, a minimum-margin threshold of 1 activated the penalty on less than 2% of future states of previously solved puzzles. A threshold of 5 remained selective, activating on roughly 1-13% depending on the initial iteration count and gap; a threshold of 10 activated on more than 95% of the longest-gap windows.

The `stay_margin_floor` follow-up therefore uses threshold 5 and weight 0.1. Both training windows use ordinary cross-entropy, including on unsolved puzzles. The added hinge loss applies only to puzzles whose originally blank cells were all correct at the first window's endpoint. For those puzzles, it penalizes the smallest future blank-cell margin when that margin falls below 5. It does not require the hidden state to stop moving.

This experiment combined three changes; it did not test the margin penalty alone. The branch inherited training on randomly sampled later iterations through step 39K. From step 39K onward it continued the first later-iteration window, added a second future cross-entropy window, and added the margin penalty. The branch loaded the same step-39K checkpoint and optimizer state as the earlier consistency comparison, then trained through 50K. Its final 1,000-puzzle monitoring score was 99.4% at 1024; the selected step-46K checkpoint was 99.5%. Full evaluation favored the final checkpoint:

| full 25K evaluation | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| margin-floor step-46K checkpoint | 95.97% | 98.88% | 98.32% | 75.11% |
| margin-floor final checkpoint | **96.26%** | **99.00%** | **98.46%** | **82.17%** |
| earlier consistency-switch final checkpoint | 96.34% | 97.52% | 88.98% | 30.20% |

Replacing the consistency term with the margin term improved later-iteration accuracy: relative to the earlier consistency switch, the final checkpoint gained 9.48 points at 2048 and 51.97 points at 4096 with little change at 128. The result does not isolate the margin term from the two later cross-entropy windows. It also did not create indefinite stability. This training objective reaches at most iteration 800, including gradient-free segments, so both the 2048 and 4096 evaluations test behavior beyond the trained iteration counts.

To reproduce the staged experiment, run these commands from the repository root as separate detached jobs:

```sh
# Produce the trial-0 source checkpoint.
modal run --detach looping/modal_stay_solved.py --arm control --full-50k --trial 0

# After the source job commits its step-39K checkpoint, add the second training window and margin penalty.
modal run --detach looping/modal_late_switch.py --mode margin-floor5
```

A matched comparison separates the individual training changes. All five trial-0 runs use random seed 20260730, the same 50K schedule, and ordinary 16-iteration cross-entropy as the starting recipe:

| training variant | only additional mechanism |
|---|---|
| original 16-iteration training | none |
| RMSNorm | recurrent-state RMSNorm |
| later-iteration training | cross-entropy on a randomly selected later window replaces ordinary cross-entropy on 20% of batches |
| consistency-only | future consistency on 20% of batches; ordinary cross-entropy remains active and no late-window cross-entropy is optimized |
| margin-only | future minimum-margin loss on 20% of batches; ordinary cross-entropy remains active and no late-window cross-entropy is optimized |

The final 1,000-puzzle probes separate the methods clearly:

| training variant | final at 128 | final at 1024 | best at 1024 | step-30K onward mean / minimum at 1024 |
|---|---:|---:|---:|---:|
| original 16-iteration training | 10.6% | 0.4% | 90.7% | 0.56% / 0.0% |
| RMSNorm | 88.2% | 89.3% | 90.3% | 88.73% / 85.6% |
| later-iteration training | **97.1%** | **98.9%** | 98.9% | 96.30% / 89.1% |
| consistency-only | 92.5% | 88.2% | 89.6% | 83.20% / 58.3% |
| margin-only | 96.4% | 98.7% | **99.3%** | **97.06%** / 81.9% |

Full 25K evaluations show where margin-only stops being sufficient:

| standalone checkpoint | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| margin-only step-49K checkpoint | 96.20% | **98.93%** | 83.62% | 19.78% |
| margin-only final checkpoint | 96.20% | 98.78% | 79.76% | 16.81% |
| later-iteration training final checkpoint | **96.47%** | 98.80% | **92.98%** | **43.70%** |

The individual-change runs are the valid comparison for asking which change prevents the original training recipe's accuracy loss. Margin protection alone prevented that loss in this run and was competitive through 1024, but training on later iterations retained substantially more accuracy beyond 1024. The combined recipe reaches 98.46% at 2048 and 82.17% at 4096, much higher than either individual result. The evidence is consistent with complementary effects: the margin penalty discourages incorrect digits from overtaking correct ones, while cross-entropy on later iterations directly trains predictions beyond the ordinary 16-iteration window.

A follow-up tested how far the margin-only recipe needs to iterate during training. The matched runs keep the random seed, 50K schedule, objective weight, and 20% auxiliary-batch probability fixed. They change only the initial gradient-free iteration counts and the gradient-free gaps between training windows, limiting the largest iteration reached to 80, 128, or 192. These limits include all gradient-free segments and are called caps in historical run names; they do not limit hidden-state magnitude. The run reaching iteration 800 above is the baseline.

| largest iteration reached during training | final at 128 | final at 1024 | best at 1024 | step-30K onward mean / minimum at 1024 |
|---:|---:|---:|---:|---:|
| 80 | 95.2% | 56.4% | 93.5% at 25K | 56.09% / 25.9% |
| 128 | 96.3% | 45.4% | **98.2% at 44K** | 80.33% / 35.0% |
| 192 | 94.5% | 9.2% | 97.1% at 37K | 59.52% / 6.8% |
| 800 | 96.4% | **98.7%** | 99.3% at 49K | **97.06% / 81.9%** |

The best checkpoints from the 128- and 192-iteration training limits also scored highly on the full evaluation:

| best checkpoint, full 25K evaluation | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| training through iteration 128, step 44K | **95.71%** | **98.18%** | **77.40%** | **22.85%** |
| training through iteration 192, step 37K | 94.10% | 96.00% | 67.71% | 10.41% |
| training through iteration 800, step 49K | 96.20% | 98.93% | 83.62% | 19.78% |

Training only through iteration 128 can therefore produce a model that solves 98.18% at iteration 1024, but none of the three shorter-iteration runs maintained its best accuracy through the end of training. Reaching iteration 800 was not necessary for a high-scoring checkpoint to exist, but was the only tested setting in this random seed that finished above 90% and stayed above 80% from training step 30K onward. The non-monotonic results at training limits of 128 and 192 also argue against treating 128 as a precise threshold without replication.
