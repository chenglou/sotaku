# Evolution-Strategies Experiments (July 2026)

ES evaluates perturbed copies of a model with forward passes and combines the perturbations according to their scores. It can optimize behavior at 1024 or 2048 iterations without storing those trajectories for backpropagation.

Current conclusions:

- ES can rescue some trained checkpoints with broken long-horizon behavior, but it cannot rescue every checkpoint that is healthy at 128 iterations.
- ES is useful for solved-and-settled training. The best settledness run scored 96.8% at 1024 and 2048 and 96.6% at 4096.
- Thirty-two independent perturbations and 16 positive/negative pairs perform similarly. Independent sampling is the default because it tests more distinct directions with simpler code.
- Pure ES has not learned 9x9 Sudoku from random initialization. Smooth losses reach uniform predictions; discrete fitness remains at chance.
- ES is not automatic polish. The latest run on an already-good late-consistency checkpoint ended below its starting score.

## Trained-Model Rescue

`exp_es_finetune.py` calibrates perturbation scale at startup, evaluates fresh train-split puzzles beyond the 2.7M training cut, standardizes population scores, and pulls weights gently toward the seed. The historical runs used 16 positive/negative direction pairs; the current default uses 32 independent directions.

| run | seed | fitness | generations | full-set result |
|---|---|---|---:|---|
| `es_ft_collapsed` | clean-A final, 5.4% at 1024 | solved | 120 | 96.2% at 1024, 81.5% at 2048 |
| `es_ft_stable` | stable final, 96.0% at 1024 | solved | 120 | 96.5% at 1024 and 2048 |
| `es_ft_hbs` | clean-A step 40K, 92.8% at 1024 | solved | 60 | 95.2% at 1024, 59.0% at 2048 |
| `es_ft_hbsb` | clean-B step 35K, 89.0% at 128 and 6.7% at 1024 | solved | 60 | 94.6% at 1024, 89.5% at 2048 |
| `es_ft_hburn` | burn-in step 25K, 744/1000 probe | solved | 120 | 82.3% at 1024, 52.8% at 2048 |
| `es_ft_r101`, `rbs`, `r3ph` | 0.3-3.0% at 1024 and poor at 128 | solved or cells | 60+60 | no solve-rate improvement |
| `es_ft_rburn` | burn-in final, 88.5% at 1024 | solved or cells | 60+60 | remained near 89% |

The first successful repairs suggested that healthy 128-iteration behavior was sufficient. A later eight-run cohort disproved that rule:

| run | best training material | pipeline outcome |
|---|---|---|
| e | final 97.1% at 1024 | direct success |
| h | final 98.1% at 1024 | direct success |
| b | step 45K at 95.5% | ES-polished to 96.1% full set |
| g | step 40K at 89.7%/128 and 48.8%/1024 | partial ES climb |
| d | step 45K at 93.9%/128 and 1.6%/1024 | flat under solved and dense fitness |
| a | step 40K at 84.4%/128 and 0.2%/1024 | flat |
| c | no checkpoint above 74.7% at 128 | no useful seed |
| f | no checkpoint above 0.9% at 128 | no useful seed |

Three of eight runs produced material above 94%, and one produced a partial rescue. Healthy short-horizon behavior and some correct 1024-iteration behavior both help, but neither predicts success or speed. Training comparisons should therefore report the yield of the complete train, select, and ES pipeline rather than assume that every failed run is repairable.

Burn-in-trained checkpoints were poor ES seeds in both attempts. Burn-in improves reliability by changing the states used for supervised training, but appears to place the weights in a region that this ES recipe cannot move far from.

## Sampling And Runtime

`exp_es_sampling_ablation.py` compared equal-cost estimators on a known-rescuable seed. Both arms used 32 evaluations per generation: either 16 directions at both signs or 32 independent positive directions.

| trial | paired full @1024 | independent full @1024 |
|---|---:|---:|
| 0 | 94.7% | 95.3% |
| 1 | 94.5% | 95.2% |
| 2 | 96.3% | 95.3% |
| mean | 95.14% | 95.23% |

All six runs were still partial at generation 60 and reached 94% or better by generation 120. On the difficult cohort-d seed, paired and independent sampling both remained near 3% after 60 generations. The experiment supports parity, not a quality advantage for either estimator.

The runtime study found two useful simplifications:

- Evaluating all 384 fitness puzzles in one batch was bit-exact and reduced an H200 generation from 117 to 65 seconds.
- Compiling a 32-iteration block reduced the same work to about 35 seconds, with 1-3% of cells changing because of numerical reduction order.

`exp_es_finetune.py` uses the compiled full-batch path for fitness and the unchanged eager path for the validation probe. A 120-generation rescue now fits in one two-hour H200 job.

## Horizon And Settledness Fitness

Training at an intermediate horizon transfers beyond that horizon, but direct long-horizon fitness remains stronger. Fitness at 256 iterations, given the 6.7% rescue seed, produced 94.1% at 256, 92.6% at 1024, and 68.1% at 2048. Direct 1024 fitness on the same seed reached 94.6% at 1024 and 89.5% at 2048.

`exp_es_settle.py` scores a puzzle only when it is solved and unchanged over the last 128 of 2048 iterations. Starting from the stable 96.5% model, 60 generations produced:

| 1024 | 2048 | 4096 |
|---:|---:|---:|
| 96.8% | 96.8% | 96.6% |

This is the strongest demonstrated use of ES for stability. Rewarding settledness produced a model that remained flat beyond the graded horizon.

The latest polish used the same solved-and-settled objective with 32 independent directions on the best late stay-consistency checkpoint:

| generation | solved | settled | both |
|---:|---:|---:|---:|
| start | 95.4% | 96.9% | 95.0% |
| 15, best | 95.9% | 97.5% | 95.8% |
| 59, final | 94.9% | 96.4% | 94.5% |

This run did not improve the seed. The first harness recorded the best score but saved only generations 19, 39, 59, and the final model. A deterministic 16-generation replay reproduced every probe and recovered generation 15. `exp_es_settle_independent.py` now saves every new best probe atomically and records both best and final model paths.

## Backprop-to-ES Handoff

One training lineage was handed to ES at progressively later checkpoints:

| backprop steps | ES generations | full @1024 | full @2048 |
|---:|---:|---:|---:|
| 0 | 60 | 0% | - |
| 5K | 120 | 69.0% | 36.9% |
| 10K | 120 | 39.7% | 0.1% |
| 20K | 60 | 84.7% | 50.0% |
| 40K | 60 | 95.2% | 59.0% |
| 50K | 120 | 96.5% | 96.5% |

The 5K and 10K checkpoints came from an unfinished 50K learning-rate schedule, so they are lower bounds on properly budgeted short backprop runs. The useful conclusion is narrower: a small backprop bootstrap creates an ES signal, while random initialization does not.

## From-Scratch ES

The from-scratch program tested several ways to provide a denser or more local signal:

| experiment | change | result |
|---|---|---|
| full model, solved or cell fitness | discrete score | stayed at chance |
| tiny 52K model | 2000 generations at horizon 16 | zero solves |
| scalar cross-entropy | smooth score | moved to uniform prediction, then stopped |
| population 256 pairs | more samples near the uniform floor | zero solves after 1000 generations |
| horizon-1 cross-entropy | removes recurrent credit assignment | pinned at `ln(9)` |
| easy-puzzle curriculum | more givens | same uniform floor |
| per-puzzle voting | vector-valued comparison | votes were nearly random |
| reward-modulated Hebbian update | local input/output traces | stayed at chance; fine-tuning moved downhill |

CDRGE was the largest direct test on the full 796,937-parameter Sudoku model. It estimates a gradient from positive and negative Rademacher perturbations, using the perturbation radius as both finite-difference scale and update size.

| CDRGE run | generations | final horizon-1 CE | cells | solved |
|---|---:|---:|---:|---:|
| 256 directions | 500 | 2.1973 | 11.50% | 0/1000 |
| 512 directions | 500 | 2.1972 | 11.51% | 0/1000 |

Both runs reached the uniform predictor. The horizon curriculum then switched from high-clue horizon-1 puzzles to horizon 4. Both population sizes diverged numerically because the fixed `0.01` update became too large at the new horizon. That curriculum result is a failed hyperparameter setting, not evidence against every CDRGE curriculum. A useful retry would recalibrate the update at each stage.

The combined evidence points to missing credit assignment. Backprop supplies a separate gradient for every predicted cell; these methods reduce a whole model rollout to one or a few scores. Increasing generations, population, smoothness, or model capacity has not replaced that information at practical cost.

Eggroll-style low-rank perturbations were discussed but not implemented. They remain distinct from the full random directions used by CDRGE and the standard ES runs.

## Growth And Other Representations

A 5K backprop bootstrap on the tiny model solved 51/1000 puzzles at 16 iterations and 228/1000 at 128. ES then moved the 128-iteration probe only into the 250-265 range.

Function-preserving growth did not help:

- Widening the feed-forward layer from 128 to 512 preserved fitness exactly, but did not change the ES slope.
- Doubling the residual stream from 32 to 64 by duplicate-and-halve preserved the function within bf16 noise, then drifted slightly downward.
- Compressing a trained width-128 model to width 64 by adjacent-channel averaging reduced every horizon to chance.

Pixel Sudoku replaced each digit with an 8x8 glyph bitmap and trained the same iterative architecture to output pixels. Iteration scaling and late-training collapse both remained, showing that neither depends on one-hot digit encoding. Solved-and-settled ES at 256 iterations improved the pixel model's 1024 probe from 22/1000 to 357/1000.

Two generated 9x9 maze tasks reached nearly perfect accuracy. They were useful architecture checks but too shallow for iteration scaling: 16 iterations already solved them.
