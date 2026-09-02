# Iteration Experiments

Experiments on running more model iterations at inference, adaptive stopping, and controlled comparisons.

This file records the original `sudoku-extreme` iteration experiments. See the [README](../README.md) for current recommended results. Older pre-`sudoku-extreme` experiments live in `../STALE_EXPERIMENTS_DOC.md`. Here, an iteration is one pass through the model's shared blocks; a training step is one optimizer update. Accuracy loss across inference iterations is distinct from accuracy loss across training checkpoints.

## Key Files

- `eval_more_iters.py` - Test model at different iteration counts (no retraining)
- `eval_confidence_stop.py` - Confidence-based and oscillation-based adaptive stopping
- `eval_fixed_point.py` - Test whether one model iteration preserves an injected correct answer
- `exp_baseline_lr2e3.py` - LR=2e-3 (**SOTA: 98.9%** at 1024 test iters)
- `exp_bs2048_baseline.py` - BS=2048, 16-iter, reverse curriculum (prev SOTA: 98.2%)
- `exp_bs2048_mixed.py` - BS=2048, 16-iter, mixed sampling (isolates curriculum effect)
- `exp_bs1024_curriculum.py` - BS=1024, 16-iter, reverse curriculum (tests smaller batch)
- `exp_32iters.py` - BS=2048, 32-iter, mixed sampling
- `exp_32iters_curriculum.py` - BS=2048, 32-iter, reverse curriculum (isolates training iter effect)
- `exp_bs2048_100k.py` - BS=2048, 100K steps with stretched LR (negative result)
- `exp_bs2048_fixedpoint.py` - 2x cross-entropy weight on already-correct cells (negative result)
- `exp_bs2048_fp_l2.py` - L2 loss toward the target on already-correct cells (negative result)
- `exp_bs2048_fp_copy.py` - Prediction-consistency loss on already-correct cells (negative result)
- `exp_bs2048_fp_gradmask.py` - Remove cross-entropy on already-correct cells (negative result)
- `exp_wider_6h.py` - d_model=192, 6 heads (tests wider model)
- `exp_wider_6h_lr2e3.py` - d_model=192, LR=2e-3 (peaks at 64 iters, collapses at 128)
- `exp_wider_6h_lowlr.py` - d_model=192, LR=1e-3 (tests if lower LR fixes wider collapse)
- `exp_smaller_3h.py` - d_model=96, 3 heads (tests smaller model)
- `exp_baseline_lr25e4.py` - LR=2.5e-3 (collapses at 128 iters)
- `exp_baseline_lr3e3.py` - LR=3e-3 (collapses at 64 iters)
- `exp_baseline_lr1e3.py` - LR=1e-3 (collapses at 128 iters)
- `exp_3phase_40k.py` - 3-phase curriculum, 40K steps (retains accuracy at 2048, with a lower best score)
- `exp_3phase_50k.py` - 3-phase curriculum, 50K steps (collapses)
- `exp_qhead.py` - Q-head learned halt signal (16 iters, negative result)
- `exp_qhead_32.py` - Q-head learned halt signal (32 iters, negative result)
- `eval_interventions.py` - Test-time interventions (damping, pred scaling, pre-norm)
- `eval_state_rms_cap.py` - Test a direction-preserving limit on recurrent-state RMS
- `modal_state_rms_cap.py` - Modal wrapper for state RMS limit sweeps
- `eval_spectral_radius.py` - Jacobian spectral radius via power iteration
- `modal_eval_interventions.py` - Modal wrapper for intervention sweeps
- `modal_spectral_stable.py` - Modal wrapper for spectral radius + stable model interventions
- `modal_eval.py` - Modal wrapper for running eval_more_iters on GPU

## Test-Time Iteration Scaling

Accuracy at various test-time iteration counts:

| Model | Notes | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| **exp_baseline_lr2e3** | **LR=2e-3 (SOTA)** | **81.8%** | **88.5%** | **92.5%** | **95.3%** | **97.3%** | **98.5%** | **98.9%** | **98.8%** |
| exp_bs2048_baseline | LR=1.5e-3 (prev SOTA) | 81.4% | 88.1% | 92.4% | 94.9% | 96.6% | 97.5% | 98.1% | 98.2% |
| exp_baseline_lr25e4 | LR=2.5e-3 | 81.8% | 87.7% | 90.7% | 84.1% | 50.5% | 4.5% | 0.1% | — |
| exp_baseline_lr3e3 | LR=3e-3 | 82.1% | 88.5% | 39.9% | 7.1% | 1.7% | 0.9% | 0.4% | — |
| exp_baseline_lr1e3 | LR=1e-3 | 80.9% | 87.2% | 90.9% | 89.8% | 80.2% | 50.2% | 20.8% | 5.3% |
| exp_wider_6h_lr2e3 | d=192, LR=2e-3 | 84.4% | 90.8% | 94.3% | 85.9% | 23.3% | 6.9% | 3.5% | — |
| exp_wider_6h | d=192, LR=1.5e-3 | 84.7% | 91.3% | 64.2% | 28.2% | 11.5% | 4.5% | 1.1% | — |
| exp_wider_6h_lowlr | d=192, LR=1e-3 | 84.1% | 90.8% | 94.5% | 94.7% | 8.2% | 0.9% | 0.1% | 0.0% |
| exp_smaller_3h | d=96, 3 heads | 76.8% | 83.0% | 86.5% | 87.8% | 87.7% | 87.4% | 86.4% | 73.2% |
| exp_3phase_40k | 3-phase, 10K/10K/20K | 80.0% | 86.8% | 91.0% | 94.0% | 95.4% | 95.9% | 95.9% | 95.9% |
| exp_3phase_50k | 3-phase, 15K/15K/20K | 81.2% | 87.6% | 34.9% | 6.2% | 3.5% | 0.7% | 0.0% | — |
| exp_faster_2drope | BS=4096 | 82.5% | 88.4% | 20.8% | — | — | — | — | — |
| exp_bs2048_mixed | mixed sampling | 81.2% | 88.3% | 92.3% | 94.9% | 96.4% | 97.0% | 97.3% | — |
| exp_bs1024_curriculum | BS=1024 | 79.7% | 85.9% | 89.5% | 91.6% | 54.8% | 1.2% | 0.0% | — |
| exp_32iters | 32-iter, mixed | 72.4% | 83.2% | 88.7% | — | 91.8% | 88.3% | — | — |
| exp_32iters_curriculum | 32-iter | — | 83.4% | 89.6% | 78.7% | 61.6% | 47.3% | 20.7% | — |
| exp_bs2048_100k | 100K steps (stretched LR) | 83.1% | 89.5% | 74.5% | 9.2% | 3.2% | 1.5% | 0.5% | — |
| exp_bs2048_fixedpoint | 2x CE on correct cells | 80.8% | 86.9% | 90.5% | 92.9% | 94.0% | 71.5% | 39.4% | — |
| exp_bs2048_fp_l2 | L2 toward target | 80.5% | 86.9% | 90.6% | 89.3% | 19.1% | 16.2% | 14.9% | — |
| exp_bs2048_fp_copy | Prediction consistency | 81.7% | 88.1% | 90.9% | 92.5% | 93.1% | 93.3% | 83.7% | — |
| exp_bs2048_fp_gradmask | Zero CE on correct cells | 3.2% | — | — | — | — | — | — | — |

## Other Results

| Experiment | Accuracy | Notes |
|---|---|---|
| BS=4096 + oscillation stop | 91.1% | Causal, deployable |
| BS=4096 + peak confidence (oracle) | 91.5% | Requires all iters retroactively |
| Q-head (16 iters) | 79.4% | Loss competition hurts main task |
| Q-head (32 iters) | 78.4% | Same issue |

## Answer Preservation And Fixed Points

### Injecting Answers And Training For 32 Iterations

These experiments tested whether injecting a correct answer or training through more iterations would help preserve solutions. Neither provided the intended improvement. The 32-iteration trainers use their own predictions as feedback, so this is not teacher forcing, which supplies target answers as the next inputs.

**Answer injection at initialization:** Supply the correct solution as one-hot prediction feedback while the hidden state is still `h_prev = initial_encoder(puzzle)`, then run one model iteration (`eval_fixed_point.py`).

| Model | Cells preserved | Puzzles perfectly preserved |
|---|---|---|
| BS=4096 baseline (16-iter) | 14.4% | 0/25000 |
| 32-iter mixed | 47.6% | 0/25000 |
| exp_baseline_lr2e3 (Mar 17 recheck) | 16.9% | 0/25000 |
| exp_bs2048_mixed (Mar 17 recheck) | 32.1% | 4/25000 |

The model loses the injected correct digits when its hidden state has not yet gone through any iterations. This does not test the states reached during ordinary solving, which have already undergone many updates.

**Answer preservation after many iterations:** Compare predictions at every iteration from 1022 through 1026. In this test, 24,513 of 25,000 puzzles had correct, identical predictions at all five iterations. This shows preservation over that measured interval, not an unchanged hidden state or a guarantee of indefinite preservation.

**32-iteration training with intermediate supervision:** Apply cross-entropy after each of 32 model iterations. The hypothesis was that puzzles solved by about iteration 10 would contribute another 20+ iterations of training on already-correct predictions. Feedback still comes from the model, not the answer key.

Neither 32-iteration model matched the strongest 16-iteration-trained models at large inference iteration counts. The tables above show the measured decline for each run. Extra intermediate supervision did not ensure that correct predictions would remain correct.

**Conclusion:** The initialization test and the later-state test measure different behavior. A model can fail to preserve an injected answer initially yet preserve answers it reaches after many iterations. Neither experiment establishes a hidden-state fixed point, and these results do not disprove teacher forcing as a training method.

### Checking For A Hidden-State Fixed Point

Unchanged predicted digits do not mean the continuous hidden state reaches an equilibrium of the kind used by a deep equilibrium model (DEQ). `eval_deq_root.py` measured the full update `F(h) - h` in FP32 on 1,000 held-out puzzles, stratified evenly across rating buckets, using the 98.9% `model_baseline_lr2e3.pt` checkpoint.

| Iteration | Puzzles solved | Hidden-state RMS | RMS of `F(h) - h` | Predictions unchanged one step later |
|---:|---:|---:|---:|---:|
| 16 | 82.1% | 24.2 | 1.46 | 82.1% |
| 64 | 93.1% | 74.4 | 0.856 | 93.4% |
| 128 | 95.4% | 123 | 0.733 | 95.9% |
| 1024 | 98.4% | 706 | 0.631 | 99.3% |

The hidden-state magnitude grows approximately linearly while its absolute step size plateaus near 0.63. The small relative residual at iteration 1024 (`0.000894` median) is therefore caused by the growing denominator, not convergence of the hidden state.

Two independent root solvers were given 64 function evaluations from warm states at iterations 16, 64, 128, and 1024. Anderson acceleration reported strict convergence for only 1.1–2.2% of puzzles and drove the median hidden-state RMS to `1e5`; Broyden reported 2.4–8.3% and drove it to `1e8`. At those magnitudes, FP32 rounds away the model's update and can produce a numerical `F(h) == h`; these are not useful equilibria. Neither solver accelerated the warm prediction toward the iteration-1024 answer.

**Conclusion:** The original checkpoint often preserves its predicted answer, but this test found no useful hidden-state equilibrium. Its residual state keeps accumulating updates after the answer stops changing. Using it as a DEQ would require changing and retraining the update to support convergence; implicit differentiation cannot simply be attached to the existing checkpoint.

### Reading Answers From State Direction

`eval_ray_dynamics.py` compared three same-architecture checkpoints on the same 1,000-puzzle FP32 sample through 2,048 iterations: the original 98.9% backpropagation model, the `clean-A` model with 5.4% at 1024, and that failing model after ES fine-tuning to 96.2%. For each cell state `h`, the diagnostic also evaluated the output head on `h / ||h||` with its bias omitted. This readout uses state direction alone. Neither normalization nor bias removal affects the recurrent trajectory; both apply only to the diagnostic readout.

Direction-only accuracy matched ordinary accuracy to within 0.3 percentage points at every measured iteration count for all three models, and normally matched exactly. The growing magnitude was therefore not needed to recover almost all of those predicted digits. Successive updates were also almost perfectly aligned locally after iteration 16 (`cos(delta_t, delta_{t+1}) > 0.999`), including in the failing model; smoothness between adjacent steps does not distinguish accurate from inaccurate predictions. The direction-only margin below is the correct digit's logit minus the largest incorrect logit in this normalized, bias-free readout.

| Model | @128 | @512 | @1024 | @2048 | 10th-percentile direction-only margin @1024 | Update cosine, iter 128 vs 2048 |
|---|---:|---:|---:|---:|---:|---:|
| Naturally stable BP | 95.4% | 98.0% | 98.4% | 98.7% | +0.0698 | 0.9948 |
| Collapsed before ES | 93.9% | 83.7% | 7.3% | 2.6% | -0.0327 | 0.4479 |
| After ES fine-tuning | 94.2% | 97.4% | 97.8% | 84.3% | +0.0536 | 0.9172 |

Direction changes across distant iterations distinguish these three checkpoints. The accurate original model's iteration-128 update already points almost exactly along its iteration-2048 update. The failing model moves smoothly but follows a broad curve: its lower-percentile direction-only margin crosses zero by iteration 512, after which many states favor incorrect digits. ES changes the starting weights by only 0.30% in relative L2 (parameter cosine 0.999995), yet that small change compounds over recurrence: the cosine between updates before and after ES falls from 0.944 at iteration 128 to 0.678 at 1024 and 0.484 at 2048.

The repaired model's remaining 2048 weakness has the same explanation. Its median directional margin remains positive, but the 10th percentile moves from +0.0536 at iteration 1024 to -0.0024 at 2048, exactly where accuracy falls. The naturally stable model's 10th percentile remains near +0.070 throughout.

**Conclusion:** Over the measured iterations, state direction explains the predicted digits while hidden-state magnitude keeps growing. ES changes the trajectory so correct digits remain preferred for longer; it does not stop norm growth or create a fixed point. In this three-model comparison, the accurate original model has a straighter trajectory and a wider direction-only margin than the ES-fine-tuned model. Later RMSNorm and later-iteration training comparisons show that total rotation is not a universal stability measure; minimum correct-answer margin is a more direct measure of impending prediction errors.

### Loss Changes To Preserve Correct Predictions

Four training modifications were tested. All reduced accuracy at large inference iteration counts relative to the baseline:

1. **Preservation weighting** (exp_bs2048_fixedpoint) — 2x CE weight on cells correct at previous iteration. Collapses at 512 iters.
2. **L2 toward target** (exp_bs2048_fp_l2) — MSE between softmax and one-hot target on correct cells. Collapses at 128 iters.
3. **Self-consistency copy** (exp_bs2048_fp_copy) — MSE between softmax at iter t and t-1 on correct cells. Gentlest; collapses at 1024 iters (83.7%).
4. **Gradient masking** (exp_bs2048_fp_gradmask) — zero CE on correct cells. Catastrophic failure (3.2%).

The baseline without these modifications reaches 98.1% at 1024 iterations. These experiments modify losses on predictions; they do not penalize hidden-state motion or explicitly seek `F(h) = h`.

### 100K Training Steps — LR Schedule Confound

Training for 100K steps with cosine decay stretched over 100K (exp_bs2048_100k) collapses at 64 test iters. The step-50K checkpoint already collapses — confirming the cause is the stretched LR schedule (LR still high at step 50K), not overtraining. The baseline's 50K cosine fully anneals by training end, enabling the flat minimum.

## Settings In The Early Successful Runs

The early successful checkpoints used the following settings. These comparisons concern the original training recipe; they do not establish necessary conditions for every architecture or later training method.

1. **LR in a narrow band** — for d=128, only LR=1.5e-3 to 2e-3 works. Both higher (2.5e-3, 3e-3) and lower (1e-3) collapse. The optimum is sharp at 2e-3. Higher LR causes oscillatory collapse; lower LR reaches a worse long-horizon answer trajectory.
2. **BS=2048** — BS=4096 collapses at 48 iters, while BS=1024 collapses at 256. Gradient noise or minimum geometry may explain the difference, but these experiments did not isolate the cause.
3. **Small enough model** — d=128 scales to 1024+. d=192 collapses at every LR tested. d=96 peaks early and slowly degrades. The d=192 spectral radius rebounds near its collapse point, but that correlation does not by itself explain why width hurts.
4. **Full LR annealing** — cosine schedule must decay to near-zero by training end. Stretched schedules (100K steps) or redistributed phase durations collapse because LR is still high late in training.
5. **16 supervised training iterations**: the 32-iteration runs did not match the strongest 16-iteration runs at large inference iteration counts. This comparison changes the differentiable training length, unlike later experiments that run initial iterations without gradients.
6. **No added prediction-preservation loss**: all four tested modifications (preservation weighting, L2 toward target, self-consistency, and gradient masking) hurt. They modify prediction losses, not the hidden-state fixed-point condition.

The empirical recipe is narrow: batch size, learning rate, model width, and annealing all affect whether the long trajectory retains correct-answer margin. These experiments do not establish a flat-minimum or hidden-state convergence mechanism.

**Important caveat (July 2026):** these settings are not sufficient for reliable training. See the reproduction results below: even with the same settings, only a fraction of fresh runs retain high accuracy at 1024 iterations.

## Reproducibility (July 2026)

A reproduction attempt on external GPUs collapsed at 1024 test iterations, which triggered a systematic study: 12 fresh runs of the two configs known to work (exp_baseline_lr2e3 at LR=2e-3 and exp_bs2048_baseline at LR=1.5e-3), across two platforms (Modal H200, Viridian B200), two training-loop implementations (the canonical `train()` and an independently recreated loop, proven mathematically identical in distribution), seeded and unseeded, interrupted-and-resumed and uninterrupted. Training data was digest-verified identical, and the Modal runs used the same cached image (torch 2.10.0) as the February originals.

Run convention: the July repeats ran through per-run copies of the base config files (exp_baseline_lr2e3_clean_a.py, exp_bs2048_baseline_rerun.py, and so on) because log and checkpoint filenames derive from module constants. The copies were byte-identical to their base files apart from those names and have been deleted; artifacts on the sudoku-outputs Modal volume keep the per-run names, and evaluating them works with the base config module (the architecture is identical).

Results at 1024 test iterations (all runs scored ~81% at 16 iterations, which did not distinguish their later accuracy):

| run | config | platform | uninterrupted? | @1024 |
|---|---|---|---|---|
| Feb 2026 originals | lr2e3 / bs2048 / mixed / 3phase_40k | H200 | yes | **98.9 / 98.1 / 97.3 / 95.9** |
| Mar 17 (undocumented until now) | lr2e3 + mixed sampling | H200 | resumed at 35K | 1.3% |
| Jul: recreated loop | lr2e3 | B200 | resumed at 46K | 5.8% |
| Jul: canonical loop | lr2e3 | B200 | resumed at 40K | **96.0%** |
| Jul: rerun | lr2e3 | H200 | resumed at 45K | 0.2% |
| Jul: rerun | bs2048 (1.5e-3) | H200 | resumed at 35K | 0.0% |
| Jul: seeds 101/202/303 | lr2e3 | B200 | resumed at 45K | 3.0 / 0.1 / 0.0 |
| Jul: seed 101 | bs2048 (1.5e-3) | B200 | resumed at 45K | 0.1% |
| Jul: clean A / B | lr2e3 | H200 | yes | 5.4 / **92.3** |
| Jul: clean A / B | bs2048 (1.5e-3) | H200 | yes | 31.2 / 1.2 |
| Jul: clean A / B / C | 3phase_40k | H200 | yes | 0.3 / 1.4 / 24.6 |
| Jul: clean A / B / C | bs2048_mixed (1.5e-3) | H200 | yes | 88.8 / 72.1 / **95.5** |

The recreated and canonical loops both produced successes and failures. Seeded and unseeded runs, interrupted and uninterrupted runs, and the two nominal GPU providers also produced both outcomes. The providers were later found to share Modal's driver 580.95.05 / CUDA 13.0 environment, so they were not independent platform tests. July produced about 2 stable runs out of 12, compared with February's 4 out of 4 (`p` about 0.008). The difference remains unexplained.

The February environment cannot be reconstructed precisely. The runs requested H200s but did not record the actual hardware, and a possible image rebuild between the four runs was noticed but not resolved. Current wrappers record `nvidia-smi`, PyTorch, and CUDA versions.

Accuracy at 1024 can change abruptly across training checkpoints while training loss remains smooth. One run moved 18% -> 92.5% -> 2.2% between steps 32K, 34K, and 36K. Another rose from 7.2% at step 40K to 96.0% at the end, while a run at 92.8% at step 40K finished at 31.2%. Selecting the best checkpoint during the late phase of learning-rate decay roughly doubled the reported success rate in the initial study.

Three reruns showed that the 40K three-phase recipe was not inherently safe: all failed at 1024 (0.3%, 1.4%, 24.6%). Mixed sampling degraded more gently across its four runs (97.3%, 88.8%, 72.1%, 95.5%), although four runs are too few to establish why.

These figures describe the original recipe, trained on iterations 1-16. The current recommendation trains on randomly sampled later iterations and is documented in [the looping notes](../looping/EXPERIMENTS_LOOPING.md). For the original recipe's historical best of 98.9%, several training runs and periodic 1024-iteration evaluation were needed; the final checkpoint was not reliably the best one.

## Stabilization Study (July 2026)

See [stabilize/EXPERIMENTS_STABILIZE.md](../stabilize/EXPERIMENTS_STABILIZE.md). EMA and feedback noise failed. Starting supervised training after 128 gradient-free iterations kept 6 of 7 final runs above the study's 80% threshold at 1024, with scores near 90% at best. Recurrent RMSNorm produced three 50K runs whose selected checkpoints scored 91.3-92.4% on the full test set.

## Evolution-Strategies Fine-Tuning (July 2026)

See [es/EXPERIMENTS_ES.md](../es/EXPERIMENTS_ES.md). ES fine-tuning improved selected failing checkpoints to 94-96%, but later runs showed that high 128-iteration accuracy is not sufficient. ES did not learn Sudoku from random initialization.

## Test-Time Interventions (No Retraining)

Test-time modifications to the forward pass to see if iteration collapse can be fixed without retraining.
Scripts: `eval_interventions.py`, `modal_eval_interventions.py`.

### LR=3e-3 model (d=128, collapses at 64 iters)

| Intervention | 16 | 32 | 64 | 128 | 256 | 1024 |
|---|---|---|---|---|---|---|
| Baseline | 82.1% | 88.5% | 39.9% | 7.1% | 1.7% | 0.4% |
| Damping α=0.9 | 81.8% | 88.1% | 44.6% | 8.3% | 1.7% | 0.5% |
| Damping α=0.8 | 81.0% | 87.5% | 52.3% | 9.5% | 2.2% | 0.5% |
| Damping α=0.7 | 80.1% | 86.8% | 67.6% | 11.8% | 2.6% | 0.6% |
| Damping α=0.5 | 76.7% | 84.5% | 88.3% | 18.9% | 4.6% | 0.6% |
| Pred_scale β=0.5 | 27.2% | 31.0% | 24.1% | 7.3% | 3.4% | 0.5% |
| Pred_scale β=0.3 | 0.5% | 0.2% | 0.2% | 0.1% | 0.0% | 0.0% |
| Pred_scale β=0.1 | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Pre_norm | 81.1% | 87.7% | 18.7% | 3.7% | 0.9% | 0.1% |

### d=192 model (LR=2e-3, collapses at 128 iters)

| Intervention | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| Baseline | 84.4% | 90.8% | 94.3% | 85.9% | 23.3% | 6.9% | 3.5% |
| Damping α=0.9 | 84.2% | 90.4% | 94.1% | 87.6% | 28.3% | 6.0% | 3.8% |
| Damping α=0.8 | 83.6% | 89.9% | 93.6% | 88.7% | 34.5% | 6.0% | 3.6% |
| Damping α=0.7 | 82.7% | 89.1% | 93.0% | 90.0% | 45.3% | 7.5% | 4.0% |
| Damping α=0.5 | 79.8% | 86.8% | 91.0% | 91.3% | 70.1% | 12.5% | 4.6% |
| Pred_scale β=0.5 | 66.6% | 76.8% | 82.5% | 80.4% | 51.2% | 44.7% | 35.5% |
| Pred_scale β=0.3 | 27.6% | 31.4% | 33.2% | 31.8% | 26.2% | 24.6% | 24.0% |
| Pred_scale β=0.1 | 4.5% | 4.8% | 4.9% | 4.9% | 4.6% | 4.5% | 4.2% |
| Pre_norm | 83.6% | 89.5% | 80.2% | 30.7% | 6.1% | 0.3% | 0.0% |

### Stable model (LR=2e-3, d=128 — SOTA)

| Intervention | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| Baseline | 81.8% | 88.5% | 92.5% | 95.3% | 97.3% | 98.5% | 98.9% |
| Damping α=0.9 | 81.6% | 88.1% | 92.3% | 95.0% | 96.9% | 98.2% | 98.7% |
| Damping α=0.8 | 81.0% | 87.5% | 91.7% | 94.5% | 96.3% | 97.8% | 98.5% |
| Damping α=0.7 | 80.2% | 86.9% | 91.1% | 94.0% | 95.9% | 97.2% | 97.9% |
| Damping α=0.5 | 76.8% | 84.5% | 89.0% | 92.4% | 94.6% | 95.6% | 94.1% |
| Pred_scale β=0.5 | 30.5% | 33.5% | 34.8% | 35.9% | 36.7% | 36.8% | 36.2% |
| Pred_scale β=0.3 | 4.5% | 4.6% | 4.6% | 4.6% | 4.6% | 4.5% | 4.3% |
| Pred_scale β=0.1 | 1.2% | 0.8% | 0.8% | 0.8% | 0.8% | 0.8% | 0.6% |
| Pre_norm | 81.3% | 88.5% | 92.8% | 95.3% | 96.7% | 51.5% | 16.7% |

### Limiting Recurrent-State RMS

`eval_state_rms_cap.py` tested a different intervention. After every complete four-layer iteration, it measures each cell state's RMS over the 128 hidden features and rescales the state only when its RMS exceeds a fixed limit. It does not subtract the mean, change the state's direction, or add learned parameters. A balanced 1,000-puzzle sweep tested RMS limits 8, 12, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48, 64, and 128; the limit of 12 was then confirmed on the standard 25,000-puzzle evaluation.

| Checkpoint | State RMS limit | 16 | 128 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|
| Stable BP | none | 82.19% | 95.40% | 98.94% | 98.95% |
| Stable BP | 12 | 82.14% | 95.92% | 98.66% | 98.82% |
| Collapsed before ES | none | 80.31% | 93.30% | 5.61% | 3.26% |
| Collapsed before ES | 12 | 80.02% | 91.73% | **91.45%** | **91.31%** |
| Rescued after ES | none | 80.09% | 93.32% | 95.99% | 81.68% |
| Rescued after ES | 12 | 79.86% | 91.85% | 93.50% | 93.59% |

The RMS limit of 12 changes the failing checkpoint's 5.61% at 1024 iterations and 3.26% at 2048 into 91.45% and 91.31%. It leaves the accurate original checkpoint essentially unchanged. Threshold choice matters: on the 1,000-puzzle sweep, the limit of 12 was best before ES, while 24 was best after ES at 97.5% and 97.4% for 1024 and 2048 iterations. This differs from the earlier `Pre_norm` experiment, which normalized only the output head's temporary input, leaving the carried hidden state unnormalized. A later intervention separated changes in state magnitude from changes in direction: reducing magnitude growth alone made accuracy worse, while reducing direction-changing motion improved the same checkpoints. The RMS limit changes subsequent computation; the experiment does not identify large magnitude alone as the cause of failure.

### Delayed Strong Damping (July 2026)

A later sweep changed both limitations of the original damping experiment: it left iterations 1-128 completely untouched and then tested much stronger under-relaxation. On the same 25,000-puzzle collapsed checkpoint, `alpha = 0.25` after iteration 128 scored 95.66% at 1024 and 96.20% at 2048, versus 5.64% and 3.06% undamped. The stable checkpoint scored 97.00% and 97.32% with damping versus 98.85% and 98.87% undamped, so damping remains unnecessary when the original recurrence is already stable.

This does not contradict the tables above: those apply `alpha >= 0.5` from the first iteration and damage or merely delay the useful early trajectory. The delayed policy preserves the healthy 128-iteration computation before changing the late dynamics. Alpha is checkpoint-dependent: a sweep across other collapsed checkpoints needed values from 0.25 down to 0.03125, and the smallest settings often only preserved the iteration-128 answer instead of improving it. See `looping/EXPERIMENTS_LOOPING.md` for the full policy grid, cross-checkpoint tests, and per-bucket results.

### Intervention Analysis

1. **Constant damping from iteration 1 delays the accuracy drop but does not prevent it.** For LR=3e-3, alpha=0.5 recovers 88.3% at 64 iterations (vs 39.9%), roughly doubling the iteration count before the drop. For d=192, alpha=0.5 peaks at 91.3% at 128 (vs 85.9%) and holds 70.1% at 256 (vs 23.3%). These constant policies still lose accuracy at larger counts. The delayed strong-damping policy above preserved high accuracy through the tested counts on several checkpoints.

2. **Prediction scaling is destructive for LR=3e-3 but interesting for d=192.** On LR=3e-3, even β=0.5 drops accuracy from 82% to 27% at 16 iters — the feedback loop is essential. But on d=192, β=0.5 trades peak accuracy (82.5% vs 94.3% at 64) for much gentler degradation (35.5% vs 3.5% at 1024). The model becomes worse but more stable.

3. **Pre-output normalization reduces accuracy.** `Pre_norm` makes all tested models worse, including the accurate original model (96.7% at 256, then 51.5% at 512 and 16.7% at 1024). On d=192 it gives 0.0% at 1024 (vs 3.5%). This shows that inserting LayerNorm before the output head changes information needed for accurate predictions, not that the original hidden state had converged.

4. **Damping can reduce the score of an already accurate model.** With constant damping, alpha=0.9 reaches 98.7% at 1024 (vs 98.9% baseline), while alpha=0.5 peaks at 95.6% at 512 then drops to 94.1% at 1024. Delaying damping until iteration 128 reduces the loss but still trails ordinary inference for this checkpoint.

5. **Changing the late carried-state dynamics can fix collapse.** A per-token RMS cap rescues the collapsed checkpoint from 5.61% to 91.45% at 1024, while delayed damping reaches 95.66%. Component interventions show that the immediate failure is accumulated direction drift across correct-answer boundaries, not state size alone. Neither intervention is a universal checkpoint-independent rule.

## Jacobian Spectral Radius Analysis

Estimated the spectral radius (dominant eigenvalue magnitude) of the Jacobian df/dh at various operating points using power iteration with finite-difference JVP (100 power iterations, 50 puzzles, eps=1e-3). Script: `eval_spectral_radius.py`.

A fixed point is a state `h*` with `F(h*) = h*`; it does not require a zero Jacobian. The Jacobian measures how a small perturbation to the current state changes the next state. These checkpoints do not approach a finite hidden-state fixed point, so spectral radii measured along their growing trajectories are sensitivity measurements, not a fixed-point convergence test.

| Model | SR@16 | SR@32 | SR@64 | SR@128 | SR@256 | SR Trend |
|---|---|---|---|---|---|---|
| LR=2e-3 (stable) | 55.8 | 40.6 | 29.0 | 21.0 | 14.0 | Decreasing |
| LR=3e-3 (collapse@64) | 73.0 | 64.0 | 59.9 | 63.0 | 65.1 | Flat/increasing |
| LR=1e-3 (stagnation) | 35.5 | 28.8 | 27.8 | 26.9 | 26.2 | Flat (lowest) |
| d=192 (collapse@128) | 88.0 | 69.8 | 67.1 | 76.9 | 79.2 | Decreasing then increasing |

Key findings:
1. **All measured spectral radii are much greater than 1**, including the stable SOTA model at 14-56. The usual `SR < 1` fixed-point criterion cannot be applied because these measurements are taken along a moving, growing trajectory rather than at a hidden-state fixed point.
2. **The trend correlates with stability in this small comparison.** The stable model's estimate decreases from 56 to 14, while two collapsing models flatten or rebound. The estimate remains much greater than 1, so "decreasing" does not mean that the map is approaching a contraction.
3. **Magnitude is not sufficient.** LR=1e-3 has the lowest estimates, 26-36, but worse answers. A smaller worst-case local sensitivity does not guarantee a useful answer trajectory.
4. **The d=192 estimate rebounds near its collapse point.** This is a useful warning signal in that run, not proof that the rebound causes collapse.

**Implication:** Treat the spectral-radius estimates as local sensitivity diagnostics. They do not show that the hidden state converges, enters a basin of attraction, or has a nearby fixed point. Direct trajectory and answer-margin measurements are more informative for the observed collapse.

## Key Findings

1. **BS=2048 is the observed sweet spot for iteration stability** — BS=4096 collapses at 48 iters, BS=1024 collapses at 256 iters, and BS=2048 remains healthy through 2048. The experiments did not isolate why.
2. **Sampling strategy doesn't matter** — curriculum vs mixed gives near-identical results in all comparisons.
3. **32-iteration training did not match the best 16-iteration runs at large inference counts.** These runs use intermediate cross-entropy and model-generated feedback, not teacher forcing.
4. **Predictions can stay correct while the hidden state keeps changing.** In the test spanning iterations 1022 through 1026, 24,513 of 25,000 puzzles had correct, identical predictions at all five iterations. The hidden-state norm continued growing; this is neither a hidden-state fixed point nor a guarantee about later predictions.
5. **Useful answer trajectories emerge without a convergence loss** — successful models retain or improve answers over many iterations even though their hidden states keep moving.
6. **The four prediction-preservation modifications reduced accuracy at large iteration counts.** They change losses on already-correct predictions; they do not impose a finite hidden-state equilibrium.
7. **LR=2e-3 is the sharp observed optimum for iteration scaling** — at d_model=128: LR=3e-3 collapses at 64 iters, LR=2.5e-3 at 128, LR=2e-3 scales to 1024 at 98.9%, LR=1.5e-3 scales to 1024 at 98.1%, and LR=1e-3 stalls at worse long-horizon accuracy.
8. **Wider models (d=192) collapse regardless of LR** — LR=2e-3 peaks at 64 iters (94.3%, better per-iteration than d=128's 92.5%) but collapses at 128. LR=1e-3 collapses at 256, LR=1.5e-3 at 64. The spectral radius rebounds past the collapse point (67→79 at iters 64→128), confirming the wider model's dynamics destabilize rather than converge.
9. **3-phase curriculum works if you keep phase durations** — dropping Medium+ with original durations (40K total) is stable at 95.9%, but redistributing to maintain 50K steps collapses because the LR schedule decays slower.
10. **The smaller model (d=96) peaks early then loses accuracy**: 87.8% at 128 iterations, falling to 73.2% at 2048. This does not by itself establish insufficient capacity as the cause.
11. **Q-head (learned halt) failed** — loss competition degrades main task.
12. **Test-time carried-state interventions can fix collapse.** On 25,000 puzzles, cap 12 rescues one collapsed checkpoint to 91.45% at 1024 and 91.31% at 2048. Delayed damping after iteration 128 improves the same checkpoint further, to 95.66% and 96.20%. Constant damping from iteration 1, prediction scaling, and pre-output LayerNorm still fail.
13. **Jacobian spectral radius is much greater than 1 for every measured model, including stable ones** — estimates range from 14 to 88. A decreasing estimate correlates with stability in the original comparison, but does not establish fixed-point convergence. Later causal interventions identify loss of correct-answer margin from accumulated directional drift as the immediate collapse mechanism.
14. **Historical released checkpoint: 98.9%** at 1024 test iterations with LR=2e-3 (`exp_baseline_lr2e3`), retaining 98.8% at 2048. The current recommendation trains on later iterations with a gradient-free initial segment; see [the looping notes](../looping/EXPERIMENTS_LOOPING.md).
