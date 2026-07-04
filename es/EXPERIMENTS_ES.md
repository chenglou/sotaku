# Evolution-Strategies Fine-Tuning (July 2026)

Forward-only fine-tuning of trained checkpoints on the 1024-iteration solve rate. The single script is es/exp_es_finetune.py (originally under iters/); `fitness_dense` selects cell-level or solved-puzzle fitness, `total_steps` the generation count.

Run convention: each launched run used a per-run copy of the base file (exp_es_ft_hbs.py, ...) because output filenames derive from module constants; the copies have been deleted (byte-identical apart from names, fitness mode, and generation count). The full run ledger:

| run | seed | fitness | gens | outcome @1024 |
|---|---|---|---|---|
| es_ft_collapsed | lr2e3 clean A final (5.4%) | solved | 120 | **96.2%** full set (81.5% @2048) |
| es_ft_stable | canonical B200 final (96.0%) | solved | 120 | **96.5%** full set (96.5% @2048) |
| es_ft_r101 / r1012 | seed-101 run final (3.0%) | solved / cells | 60+60 | flat — unrescuable |
| es_ft_rbs / rbs2 | bs2048 clean B final (1.2%) | solved / cells | 60+60 | flat — unrescuable |
| es_ft_r3ph / r3ph2 | 3phase clean A final (0.3%) | solved / cells | 60+60 | flat — unrescuable |
| es_ft_rburn / rburn2 | burn-in 50K final (88.5%) | solved / cells | 60+60 | ~89% — no gain |
| es_ft_hbs | bs2048 clean A step-40K checkpoint (92.8%) | solved | 60 | **95.2%** full set (59.0% @2048) |
| es_ft_hbsb | bs2048 clean B step-35K checkpoint (6.7%, 89.0% @128) | solved | 60 | **94.6%** full set (89.5% @2048) |
| es_ft_hburn | burnin128 C step-25K checkpoint (744/1000 probe) | solved | 120 | 82.3% full set (52.8% @2048) |


Backprop cannot reach the deployment horizon: differentiating through 1024 iterations is memory-impossible, and with the Jacobian spectral radius measured at 14-88 the gradients would explode into noise anyway. Evolution strategies need neither — perturb the weights, count solved puzzles, move toward the perturbations that scored best. Following the argument in apaz.dev's "Scaling To Unfathomable Depth" (the projection noise floor scales with parameter count, and at 800K parameters we are far below the regime where ES fine-tuning is known to work), exp_es_finetune.py fine-tunes a trained checkpoint directly on the 1024-iteration solve rate: 16 antithetic perturbation pairs per generation, perturbation scale calibrated at startup, rank-weighted updates, weight decay anchored to the seed, fitness on fresh train-split puzzles beyond the 2.7M training cut (rotated per generation, same slice for all members).

Two pilots, 120 generations each (~2 B200-hours per 60-generation job, chained across the platform's 2-hour limit):

| seed | @1024 before | @1024 after | @2048 after |
|---|---|---|---|
| collapsed clean run (5.4%) | 63/1000 probe | **96.2%** full set | 81.5% |
| stable B200 canonical (96.0%) | 961/1000 probe | **96.5%** full set | **96.5%** |

The collapsed model recovered completely — 5.4% to 96.2% at 1024 iterations in roughly seven forward-only GPU-hours, past every training-time intervention tested above. The stable model gained half a point and now holds flat through 2048 iterations, beyond its tuned horizon; the repaired model softens there (81.5%), so a freshly-carved basin appears shallower past the tuned horizon than a naturally-deep one. Two measurements along the way: the fitness landscape at 1024 iterations tolerates per-weight perturbations of 3e-4 almost without loss (357/384 vs 358 unperturbed) but is destroyed at 1e-3 — a remarkably sharp cliff — and a buggy first pilot that took thousand-fold-too-large steps zeroed both seeds within one generation, which is the same cliff seen from the other side.

A follow-up batch corrected the first impression that any collapsed checkpoint is salvageable. Six more 60-generation runs — three deeper collapses (3.0%, 1.2%, 0.3% at 1024 iterations) and a burn-in model (88.5%), each run twice: once with solved-puzzle fitness and once with dense correct-cell fitness — all failed to move at the solve level. The dense runs prove the failure is not a signal problem: cell counts climbed by thousands while solved puzzles stayed flat, i.e. the optimizer was climbing a slope that does not lead to the solving region. The one successful repair (5.4% to 96.2%) worked because that seed sat near the edge of the good region; seeds deep inside a bad one have no local path out. The burn-in seed neither degraded nor meaningfully improved (~88.5% to ~89%), so starting ES from a burn-in model buys nothing over starting it from any stable model — evidence against burn-in as a preparation step for ES, though not against burn-in itself.

A third batch (July 4) tested that recovery path. First, checkpoint sweeps over the two deep-collapse lineages (full test set at 1024 iterations, every saved 5K-step checkpoint): bs2048_baseline_clean_b never exceeded 6.7% (step 35,000) and 3phase_40k_clean_a never exceeded 2.3% (step 15,000). Runs that finish near 1% do not lose a good model at the end — they never visit the good region at all, at least at 5K-step sampling (long-iteration accuracy can swing within 2K steps, so a brief good window could hide between checkpoints). The harvest statistic above therefore does not cover every failure: checkpoint selection only helps runs that actually pass through stability.

The 1024-iteration score turned out to be the wrong criterion for what ES can rescue, though. That 6.7% checkpoint scores 89.0% at 128 iterations — short horizons intact, only the long-horizon drift broken, the same shape as the previously rescued 5.4% model — while the same lineage's final weights are broken at every horizon (1.2% at 1024, ~14% at 128) and had resisted ES entirely. Seeded from the checkpoint (exp_es_ft_hbsb.py), 60 generations took it from 6.7% to 94.6% on the full set, restored the monotonic iteration profile (79.2 / 92.4 / 94.6 at 16 / 128 / 1024), and held 89.5% at 2048 — the deepest basin of any repaired model (the original 5.4% rescue holds 81.5% there). So the rescue boundary is short-horizon health, not the 1024-iteration score: ES repairs weights whose 128-iteration behavior is intact and cannot repair weights that are broken at short horizons too.

The other two arms of the batch. exp_es_ft_hbs.py, seeded from bs2048_baseline_clean_a's step-40,000 checkpoint (92.8% at 1024 inside a run that finished at 31.2%), reached 95.2% at 1024 (95.5% at 128) — the third-best model this project has produced — though it softens to 59.0% at 2048. Basin depth at 2048 runs opposite to seed quality (rescued-from-6.7%: 89.5%; polished-from-92.8%: 59.0%; the 96.5% record model: flat), which is unexplained and strengthens the case for tuning at 2048 directly. exp_es_ft_hburn.py, seeded from the soft burn-in failure's step-25,000 checkpoint (744/1000 on the ES probe), rose to ~800/1000 by generation 35 and stayed there through all 120 generations; on the full set it measured 81.8% at 1024 and 43.0% at 2048 at generation 60, and 82.3% / 52.8% at generation 120 — the extra 60 generations bought half a point. Burn-in weights are now 0-for-2 as ES seeds (the 88.5% burn-in final also refused to move), while plain-training weights with intact short horizons are 4-for-4 (5.4 to 96.2, 96.0 to 96.5, 92.8 to 95.2, 6.7 to 94.6). Whatever burn-in does to guarantee a safe landing also appears to place the weights somewhere ES cannot push far from.

Practical upshot, current form: train plain runs (no burn-in), evaluate checkpoints at long iteration counts during the anneal tail, pick any checkpoint with healthy 128-iteration accuracy, and fine-tune it with ES for 60 generations (~1.5 forward-only GPU-hours). Both failed runs given this treatment came out at 94.6% and 95.2% — how the run itself ended barely matters. Still open: tune at 2048+ (now more motivated by the basin-depth inversion), and whether ES can lift a stable model past February's 98.9%.

## How Early Can ES Take Over? (July 2026)

A ladder of runs seeded from progressively earlier checkpoints of one trajectory (bs2048_baseline_clean_a), asking at what point in first-order training the network becomes something ES can carry the rest of the way (exp_es_from0/5k/10k/20k.py, dense cell-level fitness). Full-set results, with the later rungs from the sections above for comparison:

| backprop steps before ES | ES gens | probe trajectory | @16 | @128 | @1024 | @2048 |
|---|---|---|---|---|---|---|
| 0 (near-random weights) | 60 | 0 → 0, cell fitness pinned at chance | — | — | 0% | — |
| 5,000 (a tenth) | 120 | 286 → 647, still rising | 63.1% | 75.3% | **69.0%** | 36.9% |
| 10,000 | 120 | 2 → 371, still rising | 69.1% | 78.0% | 39.7% | 0.1% |
| 20,000 | 60 | 785 → 824 | 73.6% | 90.3% | **84.7%** | 50.0% |
| 40,000 (es_ft_hbs) | 60 | 922 → 958 | 80.6% | 95.5% | **95.2%** | 59.0% |
| 50,000 + ES polish (es_ft_stable) | 120 | 961 → 965 | — | — | **96.5%** | 96.5% |

The reading: backprop's role is carrying the network into territory where ES has a slope to climb. From random weights, sixty generations never solved a single puzzle and the cell-level fitness never left chance — there is no local slope toward solving at this population size. But one tenth of the training schedule already hands ES a workable seed (286/1000, lifted to 69% and still climbing when the budget ran out), and each additional block of backprop buys a higher landing point along a smooth gradient — no cliff anywhere between 5K and 50K. The step-10K seed is the same non-monotonic mid-training turbulence seen everywhere else in this project (it probes far below the *earlier* step-5K seed) and ES recovers it too, just more slowly.

Two caveats. The early rungs' iteration profiles still peak at 128 and decay after (they gained enormously at 1024 but are not monotonic like the step-35K rescue above), and their 2048 behavior is weak — these are partial models, not finished ones. And the ladder holds population and generations fixed; whether more ES budget, larger populations, or a fitness schedule that starts at short horizons closes the remaining gap from below is untested. As it stands, the cheapest route to a strong model is still moderate backprop plus ES (20-40K steps, then fine-tune), but the zero-rung's hard failure and the 5K rung's partial success bracket where "how much backprop do we actually need" lives: more than zero, possibly much less than the full schedule.

## Performance (July 2026)

The generation loop was profiled on a real seed model (es/bench_es.py, H200, 32 members × 384 puzzles × 1024 iterations):

| variant | 32 members | fidelity vs production |
|---|---|---|
| sequential eager, 256-puzzle chunks (production) | 117.3s | — |
| sequential eager, one 384 batch | 64.8s | bit-exact (0 cells differ) |
| sequential + compiled 32-iteration block | **34.7s** | ~1-3% of cells land differently |
| batched population (vmap), eager | 96.6s | ~1-3% |
| batched population + compiled | 36.7s | ~1-3% |

Two findings shaped the fold. Removing the chunking is free — per-puzzle results don't depend on batch composition, so the counts match bit for bit. And batching the whole population, the presumed big win, adds nothing over compilation alone: the compiled kernels already saturate the GPU at batch 384, so the simpler sequential-member loop stays. The compiled path's ~1-3% cell differences are not errors — reduction-order changes compound over 1024 iterations, the same magnitude of divergence seen when moving a checkpoint between GPU models.

What shipped in exp_es_finetune.py: fitness evaluation runs the compiled full-batch path (~3x cheaper generations, 43s/generation on H200 including the probe — a 120-generation fine-tune now fits one 2-hour job); the validation probe keeps the untouched eager path so probe values stay comparable across the project's history. An 8-generation live run from the 96.0% seed validated the fold: starting probe 958/1000 (historical: 961), fitness in the historical 330-370/384 band, probe stable across generations.

Also measured but deferred: per-puzzle early exit. On the healthy seed, half of all puzzles reach their final answer by iteration 7 (mean 63 of 1024) — but 4.2% never settle at all, and that fraction matches the model's failure rate: non-convergence at the fixed point identifies exactly the puzzles the model gets wrong. Early exit would cut fitness cost several times more, but it changes what fitness measures (settled answer vs answer-at-1024), so it stays out until the compiled path's headroom is exhausted. Deployment is untouched by all of this: the model ships as plain weights evaluated at a fixed iteration count, per the no-inference-time-tricks rule.
