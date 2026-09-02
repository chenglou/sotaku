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

## Results In Progress

Seed `20260730` has completed both branches. The ordered-puzzle/horizon digest matches exactly: `d244b0ca101b5d4e6fbcc232632b39fc835f10445093b2e012ee16bd2e05e42c`. All loop parameters received 4,000 optimizer updates; the initial encoder received 3,209 because it is detached on the 791 late-state batches. The original schedule's final LR is `0.00011963356315309796` in both branches.

| Seed | Burn-in dropout | Probe at 40K | 41K | 42K | 43K | Mean | Minimum |
|---|---|---:|---:|---:|---:|---:|---:|
| 20260730 | on | 98.4% | 95.5% | 97.7% | 98.0% | 97.40% | 95.5% |
| 20260730 | off | 93.2% | 98.5% | 94.7% | 98.2% | 96.15% | 93.2% |

These are BF16 1024-iteration probes on the seed's fixed 1K monitoring sample. The repeated final-step probe is counted once. Dropout-off fails the predeclared minimum-score criterion for this pair, despite finishing 0.2 points higher. Full-set control evaluations and the second pair are still running; do not promote dropout-off or infer a general success rate from these probes.
