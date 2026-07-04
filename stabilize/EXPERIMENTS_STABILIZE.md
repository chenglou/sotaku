# Stabilization Study (July 2026, 20K-Step Testbed)

Training-time interventions against the long-iteration instability documented in iters/EXPERIMENTS_ITERS.md (Reproducibility section). Scripts live in this folder; they were originally under iters/.

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

Current training recommendation, combining everything: train exp_baseline_lr2e3 with 128-iteration burn-in and the in-training probe, keep the best-probe checkpoint. Every such run so far yields a model at 84-90% at 1024 iterations — no total losses — versus the plain recipe's ~25% chance per run of a 92-99% model and ~75% chance of a near-total loss. Choose plain-plus-retries when chasing a headline number, burn-in when one run needs to count. Open questions: whether tuning the burn-in fraction or horizon closes the ~90% plateau gap, and whether longer burn-in extends protection past 2048 iterations.

The harvest statistic, across all 27 testbed runs of the first five arms: 12 of 27 passed through a mid-training checkpoint scoring 800+, versus 4 of 27 ending there. Whatever else is true, saving checkpoints against the probe and keeping the best roughly triples the yield of usable models — this works today on any config.
