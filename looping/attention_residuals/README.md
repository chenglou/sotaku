# Attention Residuals And A Gated Message Between Iterations

Does the loop design of [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) improve Sotaku? This file was written on November 23, 2026, before any training run finished. The results are in [RESULTS.md](RESULTS.md): none of the variants improved on the baseline.

## What Inkling 2 Mini Does

Inkling 2 Mini is a 24-layer language model whose middle 12 layers run three times. Two parts of its design could apply to Sotaku. The formulas below are from the appendix of [Goodfire's analysis](https://www.goodfire.com/research/interpreting-inkling-2-mini/).

- **Attention residuals** ([Kimi Team, 2026](https://arxiv.org/abs/2603.15031)). There is no single residual stream. Layers are grouped into blocks, and each block keeps the sum of its layers' outputs. A sublayer's input is a softmax mix of the embedding and the block sums so far, each scaled to unit RMS: `x = sum_j a_j * unit(v_j)` with `a = softmax_j(q . unit(v_j) / sqrt(d))`, where `q` is a learned vector for that sublayer.
- **A gated message between passes.** A pass does not continue from the previous pass's stream. It starts again from the first block's output, and each sublayer also reads a message `m`, a softmax mix of the previous pass's two block sums. The message is mixed in as `x' = keep * x + gain * m`, with `keep = 1 - tanh(F z)`, `gain = tanh(G z)` and `z = P [unit(x); unit(m)]`. The query, key, value and feed-forward inputs each read their own message. The gate starts closed.

Sotaku's baseline does neither: one stream runs through the four layers and carries over to the next iteration unchanged, and its size grows with the number of iterations.

## Arms

| Arm | Inside an iteration | Between iterations | Parameters |
|---|---|---|---:|
| `baseline` | residual stream | the stream carries over | 796,937 |
| `attnres` | attention residuals | the stream carries over | 797,961 |
| `message` | residual stream | restart from the puzzle embedding, plus the gated message | 830,729 |
| `attnres_message` | attention residuals | restart from the puzzle embedding, plus the gated message | 831,881 |

All arms keep the baseline's four layers, prediction feedback, output head, width 128 and dropout, and for the same seed those weights start from the same values. The variants are in [model.py](model.py).

Choices we made in adapting the design:

- Two blocks of two layers, which matches Inkling's two looped blocks.
- The query, key and value inputs of a layer share one message and one gate. Inkling gives each its own. With separate messages, one training step at batch size 2048 needs more than 131 GiB of GPU memory, which does not fit on an H200 once attention residuals are added. `separate_qkv=True` in [model.py](model.py) builds that version.
- The gate's bottleneck is 8 (Inkling uses 64 at width 2304). The gates account for almost all of the extra parameters, about 4% of the model.
- Sublayers keep the baseline's LayerNorm where Inkling uses RMSNorm.
- The same weights and the same gates are used at every iteration, including the first, where the message is zero. Inkling runs three passes and Sotaku runs thousands.
- In `attnres`, the carried stream is the iteration's input plus both block sums, so the stream still grows as in the baseline.

## Runs

Three seeds per arm (20261123, 20261124, 20261125), 12 runs. Every run uses the 20K development recipe unchanged: later-iteration training, batch 2048, AdamW at 2e-3, BF16 training. Final checkpoints are scored on the frozen 25K benchmark with FP32 inference at 16, 128, 1024 and 4096 iterations.

## What Would Count As An Improvement

The same rule as the [Hyperloop study](../hyperloop/README.md#decision-rules). A variant is promising against the baseline if either:

- mean accuracy at 1024 iterations improves by at least 0.5 points, accuracy at 4096 does not fall, and at least two of three seeds improve at 1024, or
- mean accuracy at 4096 iterations improves by at least 5 points, accuracy at 1024 loses at most 0.5 points, and at least two of three seeds improve at 4096.

A promising 20K result still needs a 50K confirmation. Four gated states passed this rule at 20K and failed at 50K.

## Commands

```sh
modal run --detach looping/attention_residuals/modal_run.py::preflight
modal run --detach looping/attention_residuals/modal_run.py::train --arm attnres_message --seed 20261123
modal run --detach looping/attention_residuals/modal_run.py::evaluate --arm attnres_message --seed 20261123
python -m unittest looping.attention_residuals.test_model
python -m looping.attention_residuals.summarize   # result tables from runs/attention_residuals/eval
python -m looping.attention_residuals.gate_check  # gate openings and state size, on CPU
```
