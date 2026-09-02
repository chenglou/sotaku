# Burn-in Dropout Comparison

Protocol recorded 2026-09-02 before launching the runs. The released candidate is unchanged by this experiment.

## Question

The current late-state CE recipe leaves dropout enabled while preparing detached states. Inference disables dropout. Does removing dropout only from that preparation improve later ordinary inference, or does it remove useful regularization?

## Matched Runs

- Branch the healthy step-39000 checkpoints from seeds `20260724` and `20260730`. Their historical 1024-iteration probes were 97.7% and 96.8%.
- For each source, compare burn-in dropout on versus off through step 43000: exactly 4,000 new optimizer updates. Keep the original 50K cosine schedule, optimizer, batch size 2048, 2.7M training pool, and all saved RNG states. Dropout remains enabled in the supervised 16-iteration window.
- Keep 20% late-state batches, horizons 32/64/128/256/512, and CE only. No RMSNorm, recheck, consistency, margin loss, ES, or inference damping.
- The new runs use separate directories and verify the exact source checksums. The checked-in harness contains those hashes and verifies the step, seed, optimizer, and RNG fields.
- Verify matched sampling by comparing a rolling SHA-256 of every ordered puzzle batch and sampled late horizon. The digest is saved with each resumable checkpoint. A mismatch invalidates the claim of a matched-data comparison and must be investigated before interpreting the pair. Disabling dropout consumes fewer CUDA random numbers, so later supervised dropout masks differ. These are matched starting points, not bit-identical noise realizations. Compilation and GPU arithmetic can introduce further differences.

## Decision Rule

Evaluate final checkpoints, not only the best 1K probe. Record late probe mean/minimum and use the same frozen 25K benchmark at 128/1024/2048/4096 for both members of each pair. Also record puzzles solved and subsequently lost.

Extend both members of each pair to the original 50K endpoint only if both dropout-off seeds retain 1024 accuracy within 0.5 percentage points of their paired controls, improve 2048 accuracy by at least 1 point, and do not reduce the 1024 probe floor by more than 1 point. Preserve the 4K continuation artifacts and use new directories for extensions. A split result calls for another matched seed, not changing the recommendation. A clear negative result stops this experiment. Any conclusion from these selected healthy sources concerns continuation, not training from scratch.

The benchmark has informed earlier experiment decisions. This is a development comparison, not an untouched test-set claim.

## Numerical Follow-Up

Before any continuation's final evaluation, the fixed-weight release checks found a large difference between FP32 and BF16 at deep inference. Evaluate all four continuation finals in FP32 as a secondary check, alongside the originally specified BF16 evaluation. Keep the original BF16 decision rule intact; report any arithmetic-dependent reversal separately instead of changing the success rule after seeing results.

## Results

All four branches completed. Both pairs have exactly matching ordered-puzzle/horizon digests, retained in their [training results](../release/validation/dropout). All loop parameters received 4,000 additional optimizer updates. The initial encoder received 3,218 for seed `20260724` and 3,209 for seed `20260730`, because it is detached on their 782 and 791 late-state batches. The original schedule's final LR is `0.00011963356315309796` in all four branches.

| Seed | Burn-in dropout | Probe at 40K | 41K | 42K | 43K | Mean | Minimum |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260724 | on | 97.0% | 93.5% | 92.8% | 96.9% | 95.05% | 92.8% |
| 20260724 | off | 93.1% | 89.9% | 92.8% | 90.6% | 91.60% | 89.9% |
| 20260730 | on | 98.4% | 95.5% | 97.7% | 98.0% | 97.40% | 95.5% |
| 20260730 | off | 93.2% | 98.5% | 94.7% | 98.2% | 96.15% | 93.2% |

These are BF16 1024-iteration probes on each seed's fixed 1K monitoring sample. The repeated final-step probe is counted once. Dropout-off lowers the minimum by 2.9 and 2.3 points, failing the predeclared minimum-score criterion in both pairs. Neither pair is extended to 50K.

Full frozen 25K evaluation of the step-43000 finals, eager execution, batch size 256:

| Seed | Burn-in dropout | Precision | 128 | 1024 | 2048 | 4096 |
|---|---|---|---:|---:|---:|---:|
| 20260724 | on | BF16 | 95.952% | 96.016% | 81.132% | 26.252% |
| 20260724 | off | BF16 | 96.132% | 89.256% | 53.252% | 7.676% |
| 20260724 | on | FP32 | 96.164% | 96.824% | 95.760% | 93.160% |
| 20260724 | off | FP32 | 96.016% | 95.380% | 90.148% | 78.736% |
| 20260730 | on | BF16 | 96.052% | 98.116% | 92.936% | 43.396% |
| 20260730 | off | BF16 | 95.924% | 98.300% | 89.720% | 34.420% |
| 20260730 | on | FP32 | 96.312% | 98.892% | 98.716% | 98.340% |
| 20260730 | off | FP32 | 96.260% | 99.136% | 98.844% | 98.076% |

Dropout-off also fails the required BF16 2048 improvement in both pairs: losses of 27.880 and 3.216 points. Seed `20260724` remains worse in FP32 at 1024/2048/4096, by 1.444 / 5.612 / 14.424 points. For seed `20260730`, FP32 dropout-off gains 0.244 points at 1024 and 0.128 at 2048, but loses 0.264 at 4096. That small tradeoff does not reverse the two-seed conclusion.

Keep dropout enabled during detached burn-in. Neither pair meets the predeclared extension rule, so no 50K extensions were run. The eight final evaluations retain per-puzzle predictions, independently checked against the pinned dataset. These selected-source continuations do not establish training-from-scratch reliability.
