# Training-Window Selection Results

All nine runs finished 20K optimizer updates, followed by all 18 planned full-set evaluations. The results below use 25,000 frozen benchmark puzzles, ordinary eager FP32 inference, and no damping, search, or answer selection. The [protocol](README.md) was fixed before launch; final checkpoints are primary, not whichever checkpoint scored highest afterward.

**Confidence selection did not improve training.** Always choosing the latest window produced more consistent long-iteration behavior, but none of those models matched the strongest randomly selected runs. The public recommendation and released checkpoint are unchanged. No 50K follow-up was launched.

## Final Checkpoints

Mean accuracy across three training seeds:

| Training-window selection | 128 iterations | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| Random | 93.320% | 88.124% | 82.385% | 67.483% |
| Highest confidence | 87.956% | 77.689% | 55.031% | 37.160% |
| Always latest | 88.069% | 88.243% | 88.233% | 88.228% |

Individual runs explain the averages:

| Selection | Seed | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| Random | 20260904 | 92.816% | 71.140% | 54.020% | 10.772% |
| Random | 20260905 | 93.656% | 97.612% | 97.780% | 97.348% |
| Random | 20260906 | 93.488% | 95.620% | 95.356% | 94.328% |
| Confidence | 20260904 | 88.236% | 88.188% | 88.032% | 87.772% |
| Confidence | 20260905 | 87.776% | 56.964% | 16.536% | 4.628% |
| Confidence | 20260906 | 87.856% | 87.916% | 60.524% | 19.080% |
| Latest | 20260904 | 85.800% | 85.868% | 85.864% | 85.876% |
| Latest | 20260905 | 88.244% | 88.232% | 88.236% | 88.236% |
| Latest | 20260906 | 90.164% | 90.628% | 90.600% | 90.572% |

Confidence improved only one of three paired seeds. Its mean difference from random was **-10.435 percentage points at 1024** and **-30.323 points at 4096**, failing the predeclared criteria. Latest and random had nearly identical mean 1024 accuracy, but latest retained substantially more accuracy at 4096. That stability came with lower short-iteration accuracy and no counterpart to the two strong random runs. Three seeds do not establish a precise probability of successful training.

## Selected Checkpoints

Secondary results use the checkpoint chosen during training by the fixed 1K monitoring sample, with ties going to the earliest step. Mean full-set accuracy:

| Selection | 128 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|
| Random | 92.596% | 95.447% | 94.684% | 71.719% |
| Confidence | 87.343% | 87.369% | 84.675% | 62.971% |
| Latest | 88.027% | 88.221% | 88.095% | 88.043% |

Choosing the best monitoring checkpoint does not reverse the confidence result. Random seed 20260904 is also a warning about long-iteration behavior: its selected checkpoint scored 93.192% at 1024 but only 22.964% at 4096. Selection steps and all individual scores are preserved in the downloaded run records; the [machine-readable summary](results/summary.json) includes both evaluation groups and per-run monitoring statistics.

## Training Behavior

The 1K monitoring scores from steps 12K through 20K summarize late training. The minimum below is the worst observed checkpoint across the three seeds, not the final full-set result.

| Selection | Mean 1024 monitoring accuracy | Minimum | Mean training time |
|---|---:|---:|---:|
| Random | 84.937% | 0.0% | 3.647 h |
| Confidence | 74.844% | 13.7% | 3.660 h |
| Latest | 85.507% | 78.4% | 3.668 h |

Training time includes the shared full scans and gradient replay, but excludes compilation, monitoring, checkpoint writes, and final evaluation. It is not the cost of the existing shorter-prefix public trainer.

Early rankings reversed: random led at 3K, while confidence led at 8K when a random run temporarily scored zero at 1024. Some collapses recovered with further training; others persisted into the final model. The early snapshots were not reliable substitutes for the fixed final evaluation.

In steps 12K-20K, confidence selected starts 32/64/128/256/512 on 0.00/0.38/11.67/63.24/24.71% of late batches. It did not simply copy the latest-window control. Its selected window averaged only 0.198 percentage points below the best candidate's actual solved fraction, measured afterward using the known training solutions. Thus, the poor final result cannot simply be summarized as confidence choosing obviously bad current answers: choosing good current answers did not produce the best trained model. This diagnostic does not establish the cause of the difference or test other confidence measures or per-puzzle selection.

## Verification

The downloaded records verified identical initialization digests, puzzle-sampling digests, data checksums, source hashes, and recurrent-work counts within each seed group. All 18 inference exports passed manifest/checksum checks, matched the intended checkpoint step and unchanged model settings, used the frozen benchmark indices, and had per-puzzle counts matching their reported scores. FP32, disabled TF32 matmul, and every-iteration solution tracking were verified from evaluator metadata.

The [preflight](results/preflight.json) and 175 local tests passed before training. Compiled BF16 scan and replay are not bit-identical. During late training their mean absolute confidence difference was below 0.00027 in every run; that small observed score difference does not prove identical gradients or trajectories.

Artifacts remain under `/outputs/window_selection_v1_20260904/runs/` on the `sudoku-outputs` Modal volume. [jobs.json](jobs.json) contains the completed call IDs. The local verified download is under `runs/window_selection_v1_20260904/runs/`. The reused benchmark informed development; it is not an untouched holdout, and these runs use a matched full-scan control rather than an exact replay of earlier public training runs.
