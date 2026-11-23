# Attention Residuals And Gated Message: Results

Completed November 23, 2026. All 12 training runs (4 arms, 3 seeds, 20K steps each) and all 12 evaluations finished. One run was preempted 3 minutes in and restarted from step 0; no run was relaunched. The setup and the improvement rule are in [README.md](README.md), written before any run finished.

## Conclusions

- None of the three variants improves on the baseline, and none passes the predeclared rule. No 50K confirmation is planned.
- **Attention residuals alone are worse at every iteration count.** Mean accuracy at 1024 iterations falls from 95.17% to 82.10%, and one seed falls from 72% at 1024 iterations to 37% at 4096.
- **The gated message makes the model stable and less accurate.** Both message arms never lose a puzzle after solving it: across 128, 1024 and 4096 iterations, the number of lost puzzles is 0 for all six runs. But the two arms stop near 83–84%, about 11 points below the baseline at 1024 iterations and about 8 points below at 4096.
- The message arms match the baseline at 16 iterations (about 77%) and then gain only 6 to 7 points from more iterations, where the baseline gains 17. The single-gated-state arm of the [Hyperloop study](../hyperloop/RESULTS.md) behaved the same way (85–89%, flat).

## Final Checkpoints

Exact-puzzle accuracy on the frozen 25K sudoku-extreme development benchmark, FP32 inference, weights from step 20K. Per-run solved and lost counts are in [scores.json](scores.json).

| Arm | Seed | 16 iterations | 128 iterations | 1024 iterations | 4096 iterations |
|---|---:|---:|---:|---:|---:|
| `baseline` | 20261123 | 77.800% | 94.056% | 96.132% | 95.000% |
| `baseline` | 20261124 | 77.924% | 93.956% | 92.640% | 87.088% |
| `baseline` | 20261125 | 78.200% | 94.784% | 96.744% | 93.460% |
| `attnres` | 20261123 | 76.372% | 86.812% | 72.380% | 37.356% |
| `attnres` | 20261124 | 76.556% | 86.668% | 86.768% | 83.124% |
| `attnres` | 20261125 | 76.736% | 86.824% | 87.136% | 87.124% |
| `message` | 20261123 | 77.764% | 82.692% | 82.816% | 82.828% |
| `message` | 20261124 | 76.388% | 81.852% | 82.208% | 82.288% |
| `message` | 20261125 | 77.564% | 84.600% | 85.236% | 85.484% |
| `attnres_message` | 20261123 | 76.632% | 82.184% | 82.620% | 82.724% |
| `attnres_message` | 20261124 | 77.188% | 81.984% | 82.216% | 82.296% |
| `attnres_message` | 20261125 | 76.316% | 85.268% | 86.752% | 87.320% |

| Arm | Mean at 16 | Mean at 128 | Mean at 1024 | Mean at 4096 | Solved at 1024, then lost by 4096 (per seed, of 25,000) |
|---|---:|---:|---:|---:|---:|
| `baseline` | 77.975% | 94.265% | 95.172% | 91.849% | 292 / 1,418 / 831 |
| `attnres` | 76.555% | 86.768% | 82.095% | 69.201% | 8,759 / 916 / 4 |
| `message` | 77.239% | 83.048% | 83.420% | 83.533% | 0 / 0 / 0 |
| `attnres_message` | 76.712% | 83.145% | 83.863% | 84.113% | 0 / 0 / 0 |

Against the rule: every variant's mean at 1024 iterations is more than 11 points below the baseline's, so neither condition can hold. The best single variant run (`attnres_message`, seed 20261125, 87.32% at 4096) is still below the baseline's mean at 4096.

## What The Variants Do Inside

[gate_check.py](gate_check.py) runs the final weights for 64 iterations on 200 benchmark puzzles. Ranges below are over the three seeds.

| Arm | RMS of the carried state after 16 / 64 iterations | Solved of 200 after 16 / 64 iterations |
|---|---|---|
| `baseline` | 14–17 / 28 | 143–158 / 176–184 |
| `attnres` | 29–33 / 88–92 | 6–138 / 92–158 |
| `message` | 7.1–8.5 / 7.3–8.7 | 152–158 / 161–170 |
| `attnres_message` | 5.8–6.1 / 5.9–6.2 | 148–155 / 162–166 |

- **The message gates opened.** At iteration 64 the gated message makes up 16–52% of a layer input's norm in `message` and 19–68% in `attnres_message`. The low accuracy is therefore not a case of gates that stayed closed.
- **The message arms carry a state of constant size.** The baseline's stream keeps growing with every iteration, and the message arms' state stops growing after about 16 iterations. This matches the scores: the message arms never lose a solved puzzle, and they also stop making progress on the puzzles they have not solved.
- **Attention residuals alone grow the carried stream about three times as fast as the baseline.** In `attnres` the carried stream is the iteration's input plus both block sums, and each sublayer reads unit-size inputs, so nothing limits the growth. During training, the BF16 check at 1024 iterations solved only 27%, 6% and 4% of its 1,000 puzzles at step 20K, far below the FP32 scores above. We did not test whether the faster growth causes the gap between BF16 and FP32.

Our reading, which these runs do not test directly: the accumulating stream is what lets the baseline keep working on a hard puzzle for hundreds of iterations, and the same accumulation is what makes it lose puzzles at 4096 iterations. Restarting every iteration from the puzzle embedding removes both.

## Limits Of This Comparison

- The adaptation differs from Inkling 2 Mini in the ways listed in [README.md](README.md): one shared message for a layer's query, key and value inputs, a gate bottleneck of 8, two blocks of two layers, and LayerNorm. Inkling runs its looped layers three times in a language model; these runs say nothing about Inkling itself.
- Learning rate, batch size and schedule were tuned for the baseline and were not retuned for any variant.
- Three seeds at 20K steps. The gaps here (11 points or more at 1024 iterations) are much larger than the spread between seeds, but a longer schedule was not run.

## Cost And Artifacts

Training took 136–139 minutes per run for `baseline`, 168–172 for `attnres`, 214–220 for `message` and 241–256 for `attnres_message` on an H200, including compilation and the checks every 1,000 steps. The 12 runs used about 38 GPU-hours.

Weights, logs, evaluation records and status files are on the Modal volume `cheng-sotaku-outputs` under `attnres20k/`, `evaluations/attnres20k_*` and `status/`. A local copy is in `runs/attention_residuals/` (not in git). To rebuild the tables: `python -m looping.attention_residuals.summarize`.
