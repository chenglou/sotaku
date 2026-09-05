# Hyperloop Results

Completed September 4, 2026; downloaded and independently checked September 5. All nine exact-20K training runs and all 18 full evaluations completed without numerical failures. No replacement jobs were launched. The v2 checkpoint and public defaults are unchanged.

## Conclusions

- Four gated states improved final 1024-iteration accuracy in all three paired seeds. The mean increased from 96.392% to 98.888%, a gain of 2.496 percentage points.
- Gates on a single state performed worse: 87.545% mean accuracy at 1024. Extra streams helped in this comparison, but the four-state gate network also has more parameters; this does not isolate additional memory from gate capacity.
- Four states did not eliminate long-iteration collapse. One final model fell from 99.216% at 1024 to 76.980% at 4096. Another improved with more iterations, reaching 99.324% at 4096.
- Both baseline and four-state variants met the predeclared reliability criteria in two of three final runs. The four-state mean at 4096 is much higher partly because one baseline collapsed to 2.856%. Three seeds do not establish a precise success rate.

These results pass the [predeclared comparison](README.md#decision-rules) against both baseline and one gated state. They justify further confirmation, not a claim of guaranteed stability or an immediate default change. The protocol requires separately specified 50K confirmation before changing the release.

## Final Checkpoints

Exact-puzzle accuracy on the same frozen 25K sudoku-extreme development benchmark, using ordinary eager FP32 inference with TF32 matmul disabled. All models below received exactly 20K optimizer updates. No damping, answer selection, search, ES, or auxiliary loss was used.

| Variant | Seed | 1024 iterations | 2048 iterations | 4096 iterations |
|---|---:|---:|---:|---:|
| Baseline | 20260907 | 97.156% | 97.112% | 97.028% |
| Baseline | 20260908 | 94.312% | 93.896% | 2.856% |
| Baseline | 20260909 | 97.708% | 96.240% | 92.960% |
| One gated state | 20260907 | 85.480% | 85.508% | 85.512% |
| One gated state | 20260908 | 89.104% | 89.220% | 89.340% |
| One gated state | 20260909 | 88.052% | 88.144% | 88.188% |
| Four gated states | 20260907 | 99.216% | 98.272% | 76.980% |
| Four gated states | 20260908 | 99.096% | 99.232% | 99.324% |
| Four gated states | 20260909 | 98.352% | 98.092% | 95.180% |

| Variant | Mean at 1024 | Mean at 2048 | Mean at 4096 |
|---|---:|---:|---:|
| Baseline | 96.392% | 95.749% | 64.281% |
| One gated state | 87.545% | 87.624% | 87.680% |
| Four gated states | 98.888% | 98.532% | 90.495% |

Accuracy at 16, 128, and 512 iterations, difficulty breakdowns, and all selected-checkpoint results are in the [full report](results/full_results_20260905.json). The benchmark was reused during development; it is not an untouched test set.

## Selected Checkpoints

Selection used only 1024-iteration accuracy on the fixed 1K monitoring sample, retaining the earliest maximum. These are secondary results, not substitutes for the final-checkpoint comparison.

| Four-state seed | Selected update | Full-set 1024 | Full-set 2048 | Full-set 4096 |
|---|---:|---:|---:|---:|
| 20260907 | 17K | 99.192% | 98.260% | 79.412% |
| 20260908 | 19K | 99.148% | 99.304% | 99.396% |
| 20260909 | 20K | 98.352% | 98.092% | 95.180% |

The strongest deep-iteration checkpoint is `runs/gated_four_seed20260908/best_validation.pt`, with its adjacent manifest. The collapse in seed 20260907 remains present in both final and selected checkpoints. Choosing by the small monitoring sample did not reliably choose the highest full-set 1024 score either; for example, its selected checkpoint scored 99.192%, versus 99.216% for the final model.

From updates 12K through 20K, the minimum 1024 monitoring scores were 83.0/76.1/85.2% for baseline, 81.4/82.3/82.4% for one gated state, and 96.6/96.8/90.5% for four gated states. Better late-training scores at 1024 did not guarantee stability at 4096.

## Verification And Cost

The [artifact audit](results/artifact_audit_20260905.json) and full report verify all nine configurations, source/data hashes, identical paired base initializations, puzzle/depth sampling digests, optimizer-update budgets, earliest-best selection, export checksums, and FP32 evaluation settings. All 2.7 million saved puzzle/horizon predictions were checked against the known answers and reported scores. Final and selected exports from the same update have identical tensors and prediction arrays.

Mean optimizer-update time on H200 was approximately 123 minutes for baseline, 133 minutes for one gated state, and 146 minutes for four gated states. Four states added about 19% training time despite adding less than 1% parameters. These times exclude compilation, monitoring, checkpoint writes, and full evaluation. The preflight measured peak allocations of 50.6/55.2/59.5 GiB; those are preflight measurements, not a full-run memory profile.

Durable artifacts are on `sudoku-outputs` under `/hyperloop_v1_20260905/runs/`. [jobs.json](jobs.json) records the original job IDs. Downloaded artifacts are under `runs_modal/hyperloop_v1_20260905/`; historical studies were not modified. Do not relaunch completed jobs.
