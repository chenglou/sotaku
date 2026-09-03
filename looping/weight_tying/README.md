# Weight-Tying Study

Preregistered on 2026-09-02, after the v2 release. This study tests whether sharing transformer weights changes what training discovers, separately from parameter storage and arithmetic cost. The released model and public defaults are unchanged.

## Comparisons

Each model performs four transformer blocks per iteration and shares the input encoder, prediction-feedback projection, and output head. `tied` uses the same four blocks at every iteration. `untied_compute` has sixteen independent four-block stages at the same width. Its stages start as copies of the tied model, so the two models initially compute the same function. The copies can subsequently learn different weights. `untied_parameters` narrows the independent stages to width 32 and feedforward width 124, keeping the parameter count within 0.1% of `tied`.

The same-width pair has equal nominal transformer matmul FLOPs, depth, puzzle batches, and optimizer updates, not equal parameter count. Compiler-selected recomputation can change actual executed work. Optimizer work and memory traffic also differ, so report measured time alongside the FLOP estimates. The parameter-matched pair differs in width, positional representation, and compute; it is not an isolated weight-sharing intervention. Report both comparisons, not one as a substitute for the other.

There are two training regimes, three architectures, and three paired random seeds: 18 runs. All use the existing 20K schedule, batch size 2048, AdamW, dropout 0.1, and 16-iteration averaged cross-entropy. `early` trains iterations 1-16. `late` uses the v2 recipe: on 20% of batches, advance 32/64/128/256/512 iterations without gradients before supervising the next 16. No auxiliary losses, added recurrent normalization, ES, or inference damping are used.

An untied finite stack has no stage 17. Its primary early-training comparison ends at iteration 16. Repeating that stack beyond iteration 16 is explicitly reported as a diagnostic, not as additional independently trained layers. The late-training comparison necessarily repeats the 16-stage bank; it tests period-1 versus period-16 weight sharing, not a completely untied 528-stage network. Never label the latter comparison fully untied.

## Data And Selection

Training uses the pinned sudoku-extreme revision and its first 2.7M training rows. Models share independently seeded puzzle and horizon samplers; model size and dropout cannot change the sampled examples. The existing 25K benchmark supplies a fixed 1K validation sample. Validation is deliberately not called held-out testing.

A new 10K-puzzle test set is generated with QQWing 1.3.4, 2,500 puzzles per generator difficulty. Each puzzle must have exactly one solution. Questions and completed grids are screened against the entire pinned sudoku-extreme train and test splits, including digit relabelings and the eight rotations/reflections. This does not establish non-equivalence under every Sudoku symmetry. QQWing uses its own random generation; the accepted dataset bytes, generator version, binary hash, raw output, and checksums are frozen. Reproducing the evaluation uses those bytes rather than assuming a seed reproduces QQWing's process.

The new set is a different distribution from sudoku-extreme. Report its scores separately; it cannot replace or be pooled into the published benchmark. The generator and solver are used only for dataset construction, never to assist neural inference. No model sees the new test set until all 18 final checkpoints and validation-based selections have been fixed. Future tuning would require another untouched set.

## Recorded Outcomes

- Primary: final-checkpoint puzzle accuracy at iteration 16 for the finite-depth comparison, and at iteration 1024 for the late-training comparison.
- Secondary: final-checkpoint accuracy at 128/2048/4096, selected-checkpoint accuracy, per-difficulty scores, and paired per-puzzle differences.
- Training reliability: report every seed, including failures, along with the fraction reaching 90% validation accuracy at 1024 and losing no more than five percentage points by 4096. Also report the mean and minimum validation score from step 12K onward. Finite untied results beyond 16 are repetition diagnostics, not its primary reliability measure.
- Compute: parameters, estimated forward/backward matmul FLOPs, actual optimizer updates, processed examples, and separate preparation, compilation, training, evaluation, and checkpoint durations.

The best checkpoint uses validation accuracy at 16 for `early` and 1024 for `late`, with the earlier step winning ties. Save matching weights immediately. Final checkpoints remain primary, including when they lose accuracy late in training. Test results cannot change this selection rule.

Three seeds provide an initial comparison, not a precise success probability. Use paired seed differences and report all three rather than treating thousands of puzzles as thousands of independent training runs. Fixed v2 optimizer settings test performance under that recipe; they do not establish that either architecture is superior after equally thorough hyperparameter tuning. Any 50K replication or learning-rate study must be specified before reading a new held-out test set.

## Sources

- [Universal Transformers](https://arxiv.org/abs/1807.03819) describes recurrent weight sharing as an inductive bias.
- [QQWing](https://github.com/stephenostermiller/qqwing) supplies puzzle generation and uniqueness checks; [CLI documentation](https://qqwing.com/instructions.html) defines its difficulty labels.

## Running The Study

All commands run from the repository root after `source venv/bin/activate`. Use a separate detached invocation for every worker. Data preparation and the full-batch CUDA preflight must finish before training; training checks the preflight's source hashes before starting.

```sh
modal run --detach looping/weight_tying/modal_run.py --action prepare
modal run --detach looping/weight_tying/modal_run.py --action smoke

# Repeat this separate invocation for each architecture, regime, and seed in protocol.json.
modal run --detach looping/weight_tying/modal_run.py --action train --architecture tied --regime late --seed 20260902

# Only after all 18 training results exist and their selected weights are fixed:
modal run --detach looping/weight_tying/modal_run.py --action seal

# Then evaluate each run's final and best-validation exports separately.
modal run --detach looping/weight_tying/modal_run.py --action evaluate --architecture tied --regime late --seed 20260902 --selection final
```

Outputs are under `weight_tying_v1_20260902/` on `sudoku-outputs`. The training log records UTC timestamps and optimizer-only time separately from total training time. Every checkpoint saves optimizer, sampler and dropout RNG states, config, source hashes, and data identity. A mismatched retry is rejected instead of silently continuing another experiment.

Compilation time covers the explicit warm-up. A new compiled input signature can still add compilation time to the first training batches, so use later validation-to-validation timing differences for steady-state throughput. Total elapsed work includes preparation, compilation, training, evaluation, and checkpoint time; optimizer time is already included in training time.

Create a local destination before recursively downloading completed results with `modal volume get`. Final reports use `python -m looping.weight_tying.analyze LOCAL_STUDY_DIRECTORY --output NEW_REPORT_DIRECTORY`; `--partial` produces an explicitly incomplete progress report. The analysis verifies saved prediction checksums and independently recomputes scores. Failed runs remain in reliability denominators; paired score averages include only completed pairs and show how many pairs are missing.

Render individual-seed training curves and final-checkpoint iteration profiles with `python -m looping.weight_tying.plot REPORT_DIRECTORY`. Plotting requires Matplotlib and does not evaluate a model. Synthetic rendering fixtures are explicitly labelled and never use the locked test puzzles.
