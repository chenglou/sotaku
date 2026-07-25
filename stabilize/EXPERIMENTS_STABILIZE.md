# Stabilization Study (July 2026)

Training-time interventions against the long-iteration instability documented in iters/EXPERIMENTS_ITERS.md (Reproducibility section). Scripts live in this folder; they were originally under iters/.

Headline: EMA and feedback noise failed; 128-iteration burn-in stabilized 6 of 7 runs near a 90% ceiling; recurrent RMSNorm produced three healthy 50K runs whose harvested checkpoints scored 91.3-92.4% at 1024. RMSNorm remains the simplest bounded-state option, while randomized late-state training in `looping/` has the higher demonstrated ceiling.

Run convention: parallel runs of one config used per-run copies of the base file (exp_testbed_ema_a.py, _b.py, ...) because log and checkpoint filenames derive from module constants. The copies were byte-identical to the base apart from those names and have been deleted; recreate one by copying the base file and renaming its experiment constants. Artifacts on the sudoku-outputs Modal volume keep the per-run names.


The reproducibility study left training reliability at ~25% per run with no controllable cause. A follow-up searched for a training-time fix, using a faster experimental setup: exp_testbed_20k.py compresses the exp_baseline_lr2e3 schedule to 20K steps (~1 hour per run) and adds an in-training probe — every 1,000 steps, 128- and 1024-iteration accuracy on 1,000 fixed test puzzles. Three baseline runs confirmed the compressed schedule reproduces the instability (1 of 3 stable at the end), and the probe revealed it is far more violent than checkpoint-level sampling suggested: adjacent probes 1,000 steps apart flip between 2/1000 and 939/1000.

Five arms, all runs unseeded (probe value is 1024-iteration accuracy out of 1,000; stable = final probe ≥ 800):

| arm | change | stable at end | best probe ≥ 800 mid-run |
|---|---|---|---|
| baseline | none | 2 / 7 | 4 / 7 |
| fast EMA | weight average, ~1K-step horizon, probed separately | 1 / 6 | 3 / 6 |
| feedback noise | perturb fed-back logits (std 0.1) during training | 1 / 6 | 2 / 6 |
| slow EMA | weight average, ~10K-step horizon | 0 / 4 | 1 / 4 |
| burn-in 32 | 32 gradient-free iterations before the supervised window, 30% of steps | 0 / 4 | 2 / 4 |
| **burn-in 128** | 128 gradient-free iterations, 20% of steps | **4 / 4** | 4 / 4 |

Weight averaging failed at both horizons (the averaged weights track the live weights into collapse rather than resisting), and feedback noise did nothing. Burn-in showed a dose-response: at 32 iterations, every run kept excellent 128-iteration accuracy to the end (the only arm with no 128-iteration collapse) while 1024 stayed unprotected; at 128 iterations, all four runs ended stable at 1024. The mechanism reading: ordinary training never shows the model the states it reaches deep into its own trajectory, so behavior there is unconstrained — burn-in trains on those states directly, at forward-only cost (~20-50% extra time depending on the fraction of steps), and the protection extends several times past the trained horizon but not indefinitely.

Full 25K-puzzle evaluation of the four burn-in-128 finals: 77-78% at 16 iterations (vs ~81% baseline — a real cost), 86-89% at 128 and 1024 (flat — stable), then degradation at 2048 (44-80%). So burn-in buys horizon-bounded reliability, not the unbounded convergence the naturally-stable runs have (February's models held 98.8% at 2048).

Confirmation at the full 50K schedule (exp_lr2e3_burnin128.py, three runs): 16-iteration accuracy recovered to baseline (80.6-81.1% — the cost seen at 20K was the shorter schedule, not burn-in), and 1024-iteration results were 88.5%, 89.4%, and 40.3% — two stable, one partial failure, and even the failure is soft compared to the baseline's characteristic near-zero crashes. Pooled across both scales, burn-in stands at 6 of 7 runs stable at 1024 versus roughly 1 in 4 without it. The long-iteration plateau stays near 90% at both scales, about 6-9 points below what the rare naturally-stable runs reach, and the in-training probe rescues even the failed run (its best checkpoint scored 84.3%).

Before the recurrent-state cap experiments below, the most reliable recipe was exp_baseline_lr2e3 with 128-iteration burn-in and the in-training probe, keeping the best-probe checkpoint. Every such run yielded a model at 84-90% at 1024 iterations — no total losses — versus the plain recipe's ~25% chance per run of a 92-99% model and ~75% chance of a near-total loss. Burn-in remains useful evidence that training on deeper trajectory states improves reliability, but the cap-1 recipe below now gives better results without extra forward iterations.

The harvest statistic, across all 27 testbed runs of the first five arms: 12 of 27 passed through a mid-training checkpoint scoring 800+, versus 4 of 27 ending there. Whatever else is true, saving checkpoints against the probe and keeping the best roughly triples the yield of usable models — this works today on any config.

## Outer Recurrent-State RMS Normalization (July 2026)

The DEQ and stable-ray diagnostics showed that the SOTA model's pre-norm residual state grows without bound even after its answer settles. Two focused experiments tested whether bounding that carried state helps.

The quick inference-only diagnostic, `iters/eval_state_rms_cap.py`, applies a direction-preserving per-token RMS cap after every complete four-layer iteration. A sweep on 1,000 balanced test puzzles selected cap 12 for the collapsed clean-A checkpoint, followed by a 25,000-puzzle confirmation:

| Checkpoint | State cap | 16 | 128 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|
| Stable BP | none | 82.19% | 95.40% | 98.94% | 98.95% |
| Stable BP | 12 | 82.14% | 95.92% | 98.66% | 98.82% |
| Collapsed before ES | none | 80.31% | 93.30% | 5.61% | 3.26% |
| Collapsed before ES | 12 | 80.02% | 91.73% | **91.45%** | **91.31%** |
| Rescued after ES | none | 80.09% | 93.32% | 95.99% | 81.68% |
| Rescued after ES | 12 | 79.86% | 91.85% | 93.50% | 93.59% |

This is a real test-time rescue, not the failed pre-output LayerNorm experiment. Pre-output LayerNorm changed only the temporary tensor read by the output head. The RMS cap changes the recurrent state carried into the next iteration and preserves its direction. Cap 12 is tuned for the collapsed checkpoint; on the 1,000-puzzle sweep, the already ES-rescued model preferred cap 24 and scored 97.5% at 1024 and 97.4% at 2048.

The clean training ablation, `exp_testbed_outer_rmsnorm.py`, changes only the recurrent transition: after all four shared transformer blocks, every token is divided by its feature RMS before the state is carried forward and read by the output head. The normalization has no affine parameters, keeps the model at 796,937 parameters, and runs on every one of the 16 supervised iterations. Three explicitly seeded 20K testbed runs used the unchanged curriculum, loss, optimizer, and schedule:

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best periodic 1024 probe |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 78.4% | 87.0% | **88.5%** | 88.3% at step 18K |
| 1 | 20260721 | 77.9% | 86.6% | **88.0%** | 89.1% at step 19K |
| 2 | 20260722 | 78.5% | 87.5% | **88.2%** | 89.5% at step 16K |

All three runs finished stable, and every final 1024 score exceeded its corresponding 128 score. That is 3 of 3 stable versus 2 of 7 for the plain 20K baseline. Accuracy is in the same band as burn-in 128's four 20K runs (86-89%), but recurrent normalization needs no extra forward iterations and directly guarantees a bounded state. At this stage, recurrent normalization had only three 20K runs, so the matched cap follow-up and full-schedule comparisons below tested whether the result held at 50K.

A seed-matched follow-up separated forcing every token to RMS 1 from merely preventing its RMS from exceeding 1. The cap leaves tokens below RMS 1 unchanged, while the normalization expands them. Both runs used seed 20260720 and were otherwise identical:

| Recurrent-state operation | Final 16 | Final 128 probe | Final 1024 probe | Best periodic 1024 probe |
|---|---:|---:|---:|---:|
| Force RMS to 1 | 78.4% | 87.0% | 88.5% | 88.3% at step 18K |
| Cap RMS at 1 | 78.4% | 87.9% | 88.6% | 88.1% at step 17K |

The cap and normalization diverged from the first optimizer step, but ended effectively tied. This paired run suggested that preventing large recurrent states might be sufficient; expanding small states to RMS 1 was not required.

Two additional cap-1 runs then completed the matched 20K cohort:

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best 1024 checkpoint |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 78.4% | 87.9% | 88.6% | 88.6% at final |
| 1 | 20260721 | 78.7% | 87.6% | 88.7% | 88.7% at final |
| 2 | 20260722 | 78.4% | 87.7% | 89.1% | 89.1% at final |

Cap 1 therefore finished 3 of 3 stable at 20K, matching normalization's reliability while changing fewer states. A full 25,000-puzzle evaluation of trial 0 scored 78.34% at 16 iterations, 87.00% at 128, 87.81% at 1024, and 88.08% at 2048. Its long-horizon behavior remained flat to slightly improving outside the 1,000-puzzle probe.

The full 50K confirmation, `exp_lr2e3_outer_cap1.py`, kept the original SOTA schedule and changed only the recurrent-state cap. It probed 1,000 fixed test puzzles every 2,000 steps and saved the best 1024-iteration checkpoint:

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best 1024 probe |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 81.1% | 88.9% | **90.1%** | 91.8% at step 30K |
| 1 | 20260721 | 81.3% | 80.7% | 63.5% | **91.6% at step 38K** |
| 2 | 20260722 | 81.3% | 90.7% | **91.7%** | 92.2% at step 48K |

Cap 1 produced 2 of 3 stable final checkpoints, not 3 of 3. Trial 1 repeatedly moved between healthy and degraded long-horizon behavior, including 91.6% at step 38K and 49.6% at step 42K, even though its recurrent-state RMS could never exceed 1. Bounded state magnitude therefore does not eliminate the optimization instability. It does appear to soften it and make good checkpoints common enough to harvest: all three runs crossed 91.6% on their 1,000-puzzle probe.

The three harvested checkpoints were then evaluated on 25,000 balanced test puzzles with the same cap-1 recurrence:

| Trial | Saved step | 16 | 128 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 0 | 30K | 77.48% | 89.25% | **91.90%** | **92.37%** |
| 1 | 38K | 79.52% | 89.39% | **91.26%** | **91.69%** |
| 2 | 48K | 81.32% | 90.06% | **91.38%** | **91.62%** |

Before the full-schedule RMSNorm comparison below, the most reliable bounded-state recipe was cap 1 with the periodic long-horizon probe and best-checkpoint retention. It yielded a 91.3-91.9% model at 1024 iterations in 3 of 3 runs, with accuracy holding or improving at 2048, no extra training iterations, and no ES stage. Its final-checkpoint reliability was only 2 of 3, however. For an already-collapsed checkpoint, the cap-12 inference intervention remains immediately useful without retraining.

### Full 50K RMSNorm Confirmation

The same three seeds were then trained for 50K steps with parameter-free RMSNorm after every recurrent loop. Everything else, including the schedule, curriculum, optimizer, probe set, and best-checkpoint rule, matched the cap-1 cohort.

| Trial | Seed | Final 16 | Final 128 probe | Final 1024 probe | Best 1024 probe |
|---:|---:|---:|---:|---:|---:|
| 0 | 20260720 | 81.6% | 91.0% | **93.1%** | **93.1% at step 44K** |
| 1 | 20260721 | 81.3% | 90.0% | **91.6%** | **92.9% at step 46K** |
| 2 | 20260722 | 81.6% | 90.8% | **92.0%** | **92.5% at step 46K** |

All three final checkpoints were healthy. During the final unchanged curriculum phase, every 1024-iteration probe stayed between 88.2% and 93.1%; RMSNorm did not reproduce cap-1 trial 1's 91.6% to 49.6% collapse. RMSNorm still moved by several points earlier in training, so bounded state does not make optimization monotonic, but the late-training behavior was substantially calmer in this cohort.

Full evaluation of the harvested checkpoints on 25,000 balanced test puzzles:

| Trial | Saved step | 16 | 128 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| 0 | 44K | 81.22% | 90.86% | **92.43%** | **92.71%** |
| 1 | 46K | 80.93% | 90.56% | **92.44%** | **92.78%** |
| 2 | 46K | 81.28% | 89.92% | **91.30%** | **91.60%** |

RMSNorm beat cap 1 on two seeds and tied it on the third. Mean 1024 accuracy was 92.06% versus 91.52%, and all three models improved at 2048. Use the 50K RMSNorm recipe with a periodic 1024-iteration probe when a simple bounded recurrence matters more than the highest score.
