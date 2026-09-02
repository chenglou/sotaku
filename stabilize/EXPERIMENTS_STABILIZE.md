# Stabilization Study (July 2026)

Training changes intended to prevent accuracy loss at large inference iteration counts; see [the reproduction study](../iters/EXPERIMENTS_ITERS.md#reproducibility-july-2026). Scripts live in this folder; they were originally under `iters/`.

Headline: EMA and feedback noise failed; starting supervised training after 128 gradient-free iterations kept 6 of 7 runs at or above 80% at 1024, with scores near 90% at best. Recurrent RMSNorm produced three 50K runs whose selected checkpoints scored 91.3-92.4% at 1024. RMSNorm remains the simplest bounded-state option, while training on randomly sampled later iterations in `looping/` has achieved higher scores.

Run convention: parallel runs of one config used per-run copies of the base file (exp_testbed_ema_a.py, _b.py, ...) because log and checkpoint filenames derive from module constants. The copies were byte-identical to the base apart from those names and have been deleted; recreate one by copying the base file and renaming its experiment constants. Artifacts on the sudoku-outputs Modal volume keep the per-run names.


The reproducibility study left training reliability at ~25% per run with no controllable cause. A follow-up searched for a training-time fix, using a faster experimental setup: `exp_testbed_20k.py` compresses the `exp_baseline_lr2e3` schedule to 20K steps (~1 hour per run). Every 1,000 training steps it measures 128- and 1024-iteration accuracy on 1,000 fixed test puzzles. Historical logs call this monitoring evaluation a probe. Three baseline runs confirmed the compressed schedule reproduces the instability (1 of 3 at or above 80% at 1024 at the end). Monitoring revealed larger swings than the earlier checkpoint sampling: scores 1,000 training steps apart ranged from 2/1000 to 939/1000.

The initial five variants and the longer burn-in follow-up used no fixed random seed. Scores below count puzzles solved at iteration 1024 out of 1,000. The final-checkpoint success threshold was 800 solved puzzles. Burn-in means running the initial iterations without gradients before the supervised window.

| variant | change | final score >= 800 | best mid-training score >= 800 |
|---|---|---|---|
| baseline | none | 2 / 7 | 4 / 7 |
| fast EMA | weight average, ~1K-step horizon, probed separately | 1 / 6 | 3 / 6 |
| feedback noise | perturb fed-back logits (std 0.1) during training | 1 / 6 | 2 / 6 |
| slow EMA | weight average, ~10K-step horizon | 0 / 4 | 1 / 4 |
| burn-in 32 | 32 gradient-free iterations before the supervised window, 30% of steps | 0 / 4 | 2 / 4 |
| **burn-in 128** | 128 gradient-free iterations, 20% of steps | **4 / 4** | 4 / 4 |

Weight averaging failed with both averaging durations: averaged weights lost accuracy along with the current weights. Feedback noise did not help. With 32 initial gradient-free iterations, every run retained high 128-iteration accuracy but still lost accuracy at 1024. With 128 initial iterations, all four runs ended above the 80% threshold at 1024. Ordinary training applies loss only at iterations 1-16; burn-in adds training on later states at forward-only cost (~20-50% extra time depending on the fraction of batches). The resulting accuracy improvements extend beyond the iterations used in training, but not indefinitely.

Full 25K-puzzle evaluation of the four burn-in-128 final checkpoints: 77-78% at 16 iterations (vs ~81% baseline), 86-89% at 128 and 1024, then degradation at 2048 (44-80%). Burn-in therefore improved reliability at 1024 but not at every larger iteration count. February's successful original-model runs held 98.8% at 2048; those finite evaluations do not establish convergence or indefinitely correct answers either.

Confirmation at the full 50K schedule (`exp_lr2e3_burnin128.py`, three runs): 16-iteration accuracy recovered to baseline (80.6-81.1%), and 1024-iteration results were 88.5%, 89.4%, and 40.3%. Two passed the 80% threshold; the third lost accuracy, although less severely than the baseline's near-zero failures. Across the 20K and 50K schedules, burn-in produced 6 of 7 final checkpoints above the threshold, versus roughly 1 in 4 without it. The highest scores stayed near 90%, about 6-9 points below the successful original-model runs. Selecting an earlier checkpoint from the failed run gave 84.3%.

Before the state-magnitude experiments below, the most reliable recipe was `exp_baseline_lr2e3` with 128-iteration burn-in and selection of the best checkpoint on the monitoring sample. Every such run produced a selected model at 84-90% at 1024, versus the original recipe's ~25% chance of a 92-99% model and ~75% chance of a near-total accuracy loss. Burn-in remains evidence that training on later states improves reliability, but limiting the state RMS to 1 gave better results without extra forward iterations.

Across all 27 runs of the first five variants, 12 produced an intermediate checkpoint scoring at least 800, versus 4 ending there. Selecting the best saved checkpoint therefore roughly tripled the fraction of these runs meeting the threshold. This comparison does not establish the same improvement for every configuration.

## Normalizing The State After Each Iteration

The equilibrium and state-direction diagnostics showed that the original model's hidden-state magnitude keeps growing through the measured iterations even after its answer stops changing. Two experiments tested whether bounding the state carried between iterations helps.

The inference-only diagnostic, `iters/eval_state_rms_cap.py`, limits each cell's hidden-state RMS (root mean square) after every complete four-layer iteration. It rescales states above the limit without changing their direction; smaller states remain unchanged. A sweep on 1,000 balanced test puzzles selected an RMS limit of 12 for the failing `clean-A` checkpoint, followed by a 25,000-puzzle confirmation:

| Checkpoint | State RMS limit | 16 | 128 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|
| Stable BP | none | 82.19% | 95.40% | 98.94% | 98.95% |
| Stable BP | 12 | 82.14% | 95.92% | 98.66% | 98.82% |
| Collapsed before ES | none | 80.31% | 93.30% | 5.61% | 3.26% |
| Collapsed before ES | 12 | 80.02% | 91.73% | **91.45%** | **91.31%** |
| After ES fine-tuning | none | 80.09% | 93.32% | 95.99% | 81.68% |
| After ES fine-tuning | 12 | 79.86% | 91.85% | 93.50% | 93.59% |

Limiting the state RMS improved inference accuracy without retraining. This differs from the earlier pre-output LayerNorm experiment, which changed only the temporary tensor read by the output head. The RMS limit changes the state carried into the next iteration and preserves its direction. The value 12 was selected for the failing checkpoint; in the 1,000-puzzle sweep, the ES-fine-tuned model did best with a limit of 24, scoring 97.5% at 1024 and 97.4% at 2048.

The training comparison, `exp_testbed_outer_rmsnorm.py`, changes only the recurrent transition: after all four shared transformer blocks, each cell's state is divided by its feature RMS before being carried forward and read by the output head. The normalization has no affine parameters, keeps the model at 796,937 parameters, and runs on every one of the 16 supervised iterations. Three 20K runs with explicit random seeds used the unchanged curriculum, loss, optimizer, and schedule:

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best periodic 1024 probe |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 78.4% | 87.0% | **88.5%** | 88.3% at step 18K |
| 1 | 20260721 | 77.9% | 86.6% | **88.0%** | 89.1% at step 19K |
| 2 | 20260722 | 78.5% | 87.5% | **88.2%** | 89.5% at step 16K |

All three runs finished at or above 80% at 1024, and every final 1024 score exceeded its corresponding 128 score. That is 3 of 3 versus 2 of 7 for the original 20K baseline. Accuracy is in the same band as burn-in 128's four 20K runs (86-89%), but recurrent normalization needs no extra forward iterations and directly bounds the state magnitude. At this stage, normalization had only three 20K runs; the comparisons below test an RMS limit instead and extend both methods to 50K.

A seed-matched follow-up separated forcing every token to RMS 1 from merely preventing its RMS from exceeding 1. The cap leaves tokens below RMS 1 unchanged, while the normalization expands them. Both runs used seed 20260720 and were otherwise identical:

| Recurrent-state operation | Final 16 | Final 128 probe | Final 1024 probe | Best periodic 1024 probe |
|---|---:|---:|---:|---:|
| Force RMS to 1 | 78.4% | 87.0% | 88.5% | 88.3% at step 18K |
| Cap RMS at 1 | 78.4% | 87.9% | 88.6% | 88.1% at step 17K |

The cap and normalization diverged from the first optimizer step, but ended effectively tied. This paired run suggested that preventing large recurrent states might be sufficient; expanding small states to RMS 1 was not required.

Two additional runs with an RMS limit of 1 completed the matched 20K comparison:

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best 1024 checkpoint |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 78.4% | 87.9% | 88.6% | 88.6% at final |
| 1 | 20260721 | 78.7% | 87.6% | 88.7% | 88.7% at final |
| 2 | 20260722 | 78.4% | 87.7% | 89.1% | 89.1% at final |

The RMS limit of 1 therefore finished 3 of 3 runs at or above 80% at 1024, matching normalization while changing fewer states. A full 25,000-puzzle evaluation of trial 0 scored 78.34% at 16 iterations, 87.00% at 128, 87.81% at 1024, and 88.08% at 2048. The full-set scores also held steady or improved as inference continued.

The full 50K confirmation, `exp_lr2e3_outer_cap1.py`, kept the original schedule and changed only the state RMS limit. It evaluated 1,000 fixed test puzzles every 2,000 training steps and saved the checkpoint with the highest 1024-iteration score:

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best 1024 probe |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 81.1% | 88.9% | **90.1%** | 91.8% at step 30K |
| 1 | 20260721 | 81.3% | 80.7% | 63.5% | **91.6% at step 38K** |
| 2 | 20260722 | 81.3% | 90.7% | **91.7%** | 92.2% at step 48K |

The RMS limit of 1 produced 2 of 3 final checkpoints at or above 80% at 1024, not 3 of 3. Trial 1's accuracy varied sharply, including 91.6% at training step 38K and 49.6% at 42K, even though its recurrent-state RMS could never exceed 1. Bounded state magnitude therefore does not eliminate training instability. All three runs nevertheless produced an intermediate checkpoint scoring at least 91.6% on their 1,000-puzzle monitoring sample.

The three selected checkpoints were then evaluated on 25,000 balanced test puzzles with the same RMS limit of 1:

| Trial | Saved step | 16 | 128 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 0 | 30K | 77.48% | 89.25% | **91.90%** | **92.37%** |
| 1 | 38K | 79.52% | 89.39% | **91.26%** | **91.69%** |
| 2 | 48K | 81.32% | 90.06% | **91.38%** | **91.62%** |

Before the full-schedule RMSNorm comparison below, the most reliable bounded-state recipe used an RMS limit of 1 and selected the best checkpoint on the periodic 1024-iteration evaluation. It produced a selected model at 91.3-91.9% at 1024 in 3 of 3 runs, with accuracy holding or improving at 2048, no extra training iterations, and no ES stage. Only 2 of 3 final checkpoints passed the 80% threshold. For the tested failing checkpoint, an inference RMS limit of 12 improved accuracy without retraining.

### Full 50K RMSNorm Confirmation

The same three random seeds were then trained for 50K steps with parameter-free RMSNorm after every complete four-block iteration. Everything else, including the schedule, curriculum, optimizer, monitoring sample, and best-checkpoint rule, matched the runs with an RMS limit of 1.

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best 1024 probe |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 81.6% | 91.0% | **93.1%** | **93.1% at step 44K** |
| 1 | 20260721 | 81.3% | 90.0% | **91.6%** | **92.9% at step 46K** |
| 2 | 20260722 | 81.6% | 90.8% | **92.0%** | **92.5% at step 46K** |

All three final checkpoints scored above 90% at 1024. During the final unchanged curriculum phase, every 1024-iteration evaluation stayed between 88.2% and 93.1%; RMSNorm did not reproduce the RMS-limit trial 1's 91.6% to 49.6% drop. Scores still moved by several points earlier in training, so bounded state does not make optimization monotonic, but accuracy varied much less late in these three runs.

Full evaluation of the selected checkpoints on 25,000 balanced test puzzles:

| Trial | Saved step | 16 | 128 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 0 | 44K | 81.22% | 90.86% | **92.43%** | **92.71%** |
| 1 | 46K | 80.93% | 90.56% | **92.44%** | **92.78%** |
| 2 | 46K | 81.28% | 89.92% | **91.30%** | **91.60%** |

RMSNorm beat the RMS limit of 1 on two random seeds and tied it on the third. Mean 1024 accuracy was 92.06% versus 91.52%, and all three models improved at 2048. Use the 50K RMSNorm recipe with periodic 1024-iteration evaluation when a simple bounded recurrence matters more than the highest score.
