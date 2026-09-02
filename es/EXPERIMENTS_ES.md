# Evolution-Strategies Experiments (July 2026)

Evolution strategies (ES) evaluate perturbed copies of a model with forward passes and combine the perturbations according to their scores, also called fitness. ES can optimize behavior at 1024 or 2048 iterations without storing those trajectories for backpropagation. Monitoring uses a separate 1,000-puzzle evaluation; historical logs call that evaluation a probe.

Current conclusions:

- ES fine-tuning can improve some checkpoints whose accuracy drops at large inference iteration counts. High accuracy at 128 iterations is not enough to guarantee improvement.
- Rewarding correct predictions that match at iterations 1920 and 2048 produced a model scoring 96.8% at 1024 and 2048 and 96.6% at 4096. This objective compares two endpoints, not every prediction between them.
- Thirty-two independent perturbations and 16 positive/negative pairs perform similarly. Independent sampling is the default because it tests more distinct directions with simpler code.
- Pure ES has not learned 9x9 Sudoku from random initialization. Smooth losses reach uniform predictions; discrete fitness remains at chance.
- ES does not always improve an already accurate model. The latest fine-tuning run, starting from a checkpoint trained with an additional supervised window and consistency loss, ended below its starting score.

## Fine-Tuning Existing Models

`exp_es_finetune.py` calibrates perturbation scale at startup, evaluates fresh train-split puzzles outside the 2.7M-puzzle backpropagation pool, standardizes population scores, and penalizes movement away from the starting weights. The historical runs used 16 positive/negative direction pairs; the current default uses 32 independent directions. The names `clean-A` and `clean-B` identify uninterrupted training runs from the reproduction study.

| run | starting checkpoint | fitness | generations | full-set result |
|---|---|---|---:|---|
| `es_ft_collapsed` | clean-A final, 5.4% at 1024 | solved | 120 | 96.2% at 1024, 81.5% at 2048 |
| `es_ft_stable` | stable final, 96.0% at 1024 | solved | 120 | 96.5% at 1024 and 2048 |
| `es_ft_hbs` | clean-A step 40K, 92.8% at 1024 | solved | 60 | 95.2% at 1024, 59.0% at 2048 |
| `es_ft_hbsb` | clean-B step 35K, 89.0% at 128 and 6.7% at 1024 | solved | 60 | 94.6% at 1024, 89.5% at 2048 |
| `es_ft_hburn` | fixed burn-in training, step 25K, 744/1000 monitoring score | solved | 120 | 82.3% at 1024, 52.8% at 2048 |
| `es_ft_r101`, `rbs`, `r3ph` | 0.3-3.0% at 1024 and poor at 128 | solved or cells | 60+60 | no solve-rate improvement |
| `es_ft_rburn` | burn-in final, 88.5% at 1024 | solved or cells | 60+60 | remained near 89% |

The first successful fine-tuning runs suggested that high 128-iteration accuracy was sufficient. A later group of eight training runs disproved that rule:

| run | best checkpoint before ES | result after selection and ES |
|---|---|---|
| e | final 97.1% at 1024 | direct success |
| h | final 98.1% at 1024 | direct success |
| b | step 45K at 95.5% | ES fine-tuning reached 96.1% on the full set |
| g | step 40K at 89.7%/128 and 48.8%/1024 | some improvement with ES |
| d | step 45K at 93.9%/128 and 1.6%/1024 | flat under solved and dense fitness |
| a | step 40K at 84.4%/128 and 0.2%/1024 | flat |
| c | no checkpoint above 74.7% at 128 | no suitable starting checkpoint |
| f | no checkpoint above 0.9% at 128 | no suitable starting checkpoint |

Three of eight runs produced a checkpoint above 94%, and one produced a smaller improvement. High accuracy at 128 iterations and some correct 1024-iteration answers both help, but neither predicts success or speed. Training comparisons should therefore report the fraction of runs meeting a specified accuracy threshold after training, checkpoint selection, and ES, rather than assume that every failed run is repairable.

Checkpoints trained with a fixed number of initial gradient-free iterations, called burn-in, improved little with ES in both attempts. Burn-in changes the states used for supervised training. These attempts show limited additional benefit from this ES recipe, not that such checkpoints can never improve.

## Sampling And Runtime

`exp_es_sampling_ablation.py` compared equal-cost estimators on a starting checkpoint that had previously improved with ES. Both variants used 32 evaluations per generation: either 16 directions at both signs or 32 independent directions.

| trial | paired full @1024 | independent full @1024 |
|---|---:|---:|
| 0 | 94.7% | 95.3% |
| 1 | 94.5% | 95.2% |
| 2 | 96.3% | 95.3% |
| mean | 95.14% | 95.23% |

All six 120-generation runs reached 94% or better. Results after only 60 generations understated the later improvement. On the difficult starting checkpoint from run d, paired and independent sampling both remained near 3% after 60 generations. The experiment supports similar performance, not a quality advantage for either estimator.

The runtime study found two useful simplifications:

- Evaluating all 384 fitness puzzles in one batch was bit-exact and reduced an H200 generation from 117 to 65 seconds.
- Compiling a 32-iteration block reduced the same work to about 35 seconds, with 1-3% of cells changing because of numerical reduction order.

`exp_es_finetune.py` uses the compiled full-batch path for fitness and the unchanged eager path for monitoring. A 120-generation fine-tuning run now fits in one two-hour H200 job.

## Iteration Counts And Matching Answers

Scoring ES candidates at 256 iterations improved accuracy beyond that iteration count, but scoring directly at 1024 worked better. Starting from the checkpoint with 6.7% at 1024, fitness at 256 produced 94.1% at 256, 92.6% at 1024, and 68.1% at 2048. Direct 1024 fitness on the same starting checkpoint reached 94.6% at 1024 and 89.5% at 2048.

`exp_es_settle.py` rewards a puzzle only when all originally blank cells are correct at iteration 2048 and their predicted digits match those at iteration 1920. The saved configuration calls this `solved_and_settled`, but the implementation compares only those two snapshots: predictions may change and return in between. Starting from the model scoring 96.5%, 60 generations produced:

| 1024 | 2048 | 4096 |
|---:|---:|---:|
| 96.8% | 96.8% | 96.6% |

This is the strongest demonstrated use of ES for preserving accuracy at larger iteration counts. The full-set score remained nearly unchanged at 4096, beyond the 2048 iterations used for fitness. That observation does not establish that every puzzle stayed solved throughout the interval.

The latest fine-tuning run used the same two-snapshot objective with 32 independent directions. Its starting checkpoint came from the experiment that added a second supervised window and consistency loss at step 39K:

| generation | correct at 2048 | matching digits at 1920 and 2048 | both |
|---:|---:|---:|---:|
| start | 95.4% | 96.9% | 95.0% |
| 15, best | 95.9% | 97.5% | 95.8% |
| 59, final | 94.9% | 96.4% | 94.5% |

The final model did not improve on the starting checkpoint. The first harness recorded the best score but saved only generations 19, 39, 59, and the final model. A deterministic 16-generation replay reproduced every monitoring score and recovered generation 15. `exp_es_settle_independent.py` uses the same two-snapshot comparison, now saves every new best monitoring checkpoint atomically, and records both best and final model paths.

## When To Switch To ES

Checkpoints from progressively later steps of one backpropagation run were used as starting points for ES:

| backprop steps | ES generations | full @1024 | full @2048 |
|---:|---:|---:|---:|
| 0 | 60 | 0% | - |
| 5K | 120 | 69.0% | 36.9% |
| 10K | 120 | 39.7% | 0.1% |
| 20K | 60 | 84.7% | 50.0% |
| 40K | 60 | 95.2% | 59.0% |
| 50K | 120 | 96.5% | 96.5% |

The 5K and 10K checkpoints came from an unfinished 50K learning-rate schedule, so they do not establish the best result of a dedicated short backpropagation run. The useful conclusion is narrower: ES improved some partially trained checkpoints, while the random-initialization run did not learn to solve puzzles.

## From-Scratch ES

The from-scratch program tested several ways to provide a denser or more local signal:

| experiment | change | result |
|---|---|---|
| full model, solved or cell fitness | discrete score | stayed at chance |
| tiny 52K model | 2000 generations, scoring at iteration 16 | zero solves |
| scalar cross-entropy | smooth score | moved to uniform prediction, then stopped |
| population 256 pairs | more samples near the uniform-prediction baseline | zero solves after 1000 generations |
| cross-entropy at iteration 1 | removes credit assignment across model iterations | pinned at `ln(9)` |
| easy-puzzle curriculum | more givens | same uniform-prediction baseline |
| per-puzzle voting | vector-valued comparison | votes were nearly random |
| reward-modulated Hebbian update | local input/output traces | stayed at chance; fine-tuning moved downhill |

CDRGE was the largest direct test on the full 796,937-parameter Sudoku model. It estimates a gradient from positive and negative Rademacher perturbations, using the perturbation radius as both finite-difference scale and update size.

| CDRGE run | generations | final CE at iteration 1 | cells | solved |
|---|---:|---:|---:|---:|
| 256 directions | 500 | 2.1973 | 11.50% | 0/1000 |
| 512 directions | 500 | 2.1972 | 11.51% | 0/1000 |

Both runs reached the uniform predictor. The training schedule then switched from scoring puzzles with many givens at iteration 1 to scoring at iteration 4. Both population sizes diverged numerically because the fixed `0.01` update became too large at the new iteration count. That curriculum result is a failed hyperparameter setting, not evidence against every CDRGE curriculum. A useful retry would recalibrate the update at each stage.

The combined evidence points to limited credit assignment: determining which weight changes would improve particular predictions. Backpropagation differentiates each cell's loss through the computation; these methods reduce a whole sequence of model iterations to one or a few scores. Increasing generations, population, smoothness, or model capacity has not replaced that information at practical cost.

Eggroll-style low-rank perturbations were discussed but not implemented. They remain distinct from the full random directions used by CDRGE and the standard ES runs.

## Growth And Other Representations

Training the tiny model with backpropagation for 5K steps produced 51/1000 solved puzzles at 16 iterations and 228/1000 at 128. ES then moved the 128-iteration monitoring score only into the 250-265 range.

Function-preserving growth did not help:

- Widening the feed-forward layer from 128 to 512 preserved fitness exactly, but did not improve the rate of progress with ES.
- Doubling the residual stream from 32 to 64 by duplicate-and-halve preserved the function within bf16 noise, then drifted slightly downward.
- Compressing a trained width-128 model to width 64 by adjacent-channel averaging reduced accuracy at every tested iteration count to chance.

Pixel Sudoku replaced each digit with an 8x8 glyph bitmap and trained the same iterative architecture to output pixels. Improvement with more inference iterations and loss of accuracy late in training both remained, showing that neither depends on one-hot digit encoding. ES rewarded decoded answers that were correct at iteration 256 and matched those at iteration 192. This increased the pixel model's 1024-iteration monitoring score from 22/1000 to 357/1000; it did not check all intermediate predictions for agreement.

Two generated 9x9 maze tasks reached nearly perfect accuracy. They were useful architecture checks but too shallow for iteration scaling: 16 iterations already solved them.
