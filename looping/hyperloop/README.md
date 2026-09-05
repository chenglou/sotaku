# Shared-Gate Hyperloop Study

Controlled 9x9 Sudoku experiments, specified before training. The question is whether learned loop-level gates improve the existing v2 recipe, and whether four parallel hidden states help beyond gates on a single state. The released checkpoint and default training path are unchanged.

## Background

[Hyperloop Transformers](https://arxiv.org/html/2604.21254v2) keeps several residual streams and uses input-dependent read, write, and retention gates around a repeated transformer block. Its language-model experiments use small loop counts and loop-specific parameters. Our shared-gate variant is an adaptation for arbitrary iteration counts, not a reproduction of those experiments.

Astra's recurrent architecture is reported, not officially disclosed. [Reporting](https://www.theinformation.com/articles/secret-technique-behind-openais-astra-model-sparks-security-concerns); [OpenAI's launch description](https://openai.com/index/gpt-6-astra/), checked September 4, 2026. This is motivation for studying recurrence, not evidence that Astra uses Hyperloop or this implementation.

A directly relevant public reference is [Geiping et al., Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/html/2502.05171v1): repeated shared computation in hidden state, variable training depth, and truncated backpropagation. Those broad ideas overlap with Sotaku. They do not imply that Sudoku results transfer directly to language models.

## Architecture

The baseline calls the public model's unchanged recurrent step. Gated variants retain the same input encoder, prediction feedback, four transformer blocks, output head, width 128, and dropout 0.1. Each gated iteration reads the states into one vector, performs the existing recurrent step once, then writes its output into the parallel states. The output head sees their mean. There are still four transformer blocks, not four copies of the model.

For each cell, the implementation computes:

```text
gate_input = RMSNorm(flatten(states))  # no learned affine parameters
read, write, retain = learned_gates(gate_input)
combined = sum(read[i] * states[i])
proposal = existing_recurrent_step(combined, previous_predictions)
next_states[i] = retain[i] * states[i] + write[i] * proposal
logits = output_head(mean(next_states))
```

Only the copy supplied to the gate network is normalized. The carried states are not normalized or capped. Gates are shared across all iterations; there are no iteration embeddings or new transformer blocks. The two gated variants use the same equations.

Our read gates use `2 / streams * sigmoid(...)`, rather than the paper's unscaled sigmoid, so their initial sum is near one for both one and four streams. Write gates use `2 * sigmoid(...)`; retention uses `sigmoid(...)`. Gate projections start with learned scale 0.01, read/write biases zero, and retention bias -8. This starts near the existing update instead of immediately adding a second large residual path. It is not an exact function-preserving initialization, and a negative result would not rule out other Hyperloop initializations. Small random gate projections allow the streams to diverge from their initially identical copies.

Gate normalization and projection run in FP32; gate outputs are converted to the carried state's dtype. The rest retains BF16-autocast training. Inference uses the same equations in eager FP32. No damping, search, answer selection, extra loss, or early stopping is added.

## Matched Runs

| Arm | Parallel states | Parameters |
|---|---:|---:|
| `baseline` | 1, no gates | 796,937 |
| `gated_one` | 1, learned gates | 797,327 |
| `gated_four` | 4, learned gates | 803,096 |

Each arm has three paired seeds: 20260907, 20260908, and 20260909. Runs use 20K optimizer updates, batch size 2048, the frozen 2.7M-puzzle training pool, AdamW at 2e-3, the existing cosine schedule, and the existing reverse curriculum. The base weights are identical within each seed group. Separate random-number generators pair puzzle batches and sampled training depths independently of model size. Dropout seeds also match, although different compiled graphs need not produce identical floating-point trajectories.

Every batch trains through 16 iterations using the usual averaged cross-entropy. On 20% of batches, the model first advances 32/64/128/256/512 iterations without gradients. The other 80% start at iteration one. There is no candidate scan, confidence selection, or gradient through the initial prefix. Each arm performs the same number of transformer iterations, but gates add arithmetic and activation memory; actual time and memory are reported separately.

The 1K monitoring sample and 25K full benchmark are reused development data, not an untouched test set. Full evaluations cover 16, 128, 512, 1024, 2048, and 4096 iterations for both final and best-monitoring checkpoints. Checkpoint selection uses 1024 monitoring accuracy, with the earliest maximum retained. Final checkpoints are primary. Evaluation records individual puzzle predictions, first solution and regression counts; those diagnostics never change inference output.

## Decision Rules

All nine runs must finish with finite results. Four streams are promising relative to baseline if either:

- Mean final 1024 accuracy improves by at least 0.5 percentage points, 4096 accuracy does not decrease, and at least two of three seed pairs improve at 1024.
- Mean final 4096 accuracy improves by at least five percentage points, 1024 loses no more than 0.5 points, and at least two seed pairs improve at 4096.

Apply the same comparisons against the one-stream gated control before attributing gains to extra memory. Record every failed seed; three seeds do not establish a precise success probability. A reliable final run means at least 90% at 1024, at least 85% at 4096, and no more than a five-point decrease between them. Also report the mean/minimum 1024 monitoring accuracy from 12K onward and the complete learning curves. No release change follows without separately specified 50K confirmation.

State diagnostics measure magnitude, relative differences between streams, and gate values on the first evaluation batch. They distinguish effectively identical streams from genuinely different states; differences alone do not prove useful computation.

## Verification And Operation

The setup review found 409 parseable tracked Python files and no references to removed scripts in supported entrypoints. The preceding cleanup had already removed obsolete root scripts and centralized Modal upload exclusions; 181 core tests and 163 separate research tests passed before this study. Historical experiment code and artifacts are retained for reproducibility. New code reuses the existing transformer, sampler, learning-rate schedule, RNG helpers, and atomic checkpoint utilities instead of changing the public trainer.

CPU tests check baseline outputs, loss and gradients with dropout on/off; gate equations and gradient flow; prefix detachment; inference equivalence; exact interrupted/resumed training; selected-checkpoint recovery; and artifact identity checks. The full-batch H200 preflight tests each arm with a 512-iteration prefix, finite gradients, populated optimizer/RNG restoration, export loading, and FP32 inference. Training refuses a different source hash from the passed preflight. A launcher uploads an explicit source allowlist, not the project directory.

```sh
source venv/bin/activate
python -m unittest looping.hyperloop.test_hyperloop -v
modal run --detach looping/hyperloop/modal_run.py --action smoke

# Only after the preflight passes. Repeat as a separate invocation for each arm and seed.
modal run --detach looping/hyperloop/modal_run.py --action train --arm baseline --seed 20260907
```

Each detached worker trains and then evaluates both saved models. Artifacts live under `/outputs/hyperloop_v1_20260905/` on `sudoku-outputs`; existing study directories are not modified. `jobs.json` will record actual launches. Resumable checkpoints include optimizer, sampler and dropout RNG state, schedule/configuration, source/data hashes, and selected weights. The selected export can be reconstructed from the checkpoint after an interrupted write. Compilation warmup restores RNG and does not update weights.

Download into a new local directory and analyze with `python -m looping.hyperloop.analyze LOCAL_STUDY_DIRECTORY --data-dir PREPARED_DATA_DIRECTORY --output NEW_REPORT.json`. The analyzer checks pairing, frozen data, checkpoint steps, model and prediction checksums, and recomputes solved counts from individual predictions and known answers. Preparation, compiler warmup, monitoring, and final evaluation are not interchangeable with optimizer-update time.
