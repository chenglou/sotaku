# Positional Encoding Experiments

This document tracks all positional encoding experiments, from the original ablations through the recent sudoku-agnostic alternatives.

## Baseline: Structured row+col+box embeddings

The original architecture uses three separate `nn.Embedding(9, d_model)` tables — one each for row, column, and 3x3 box index. These are summed and added to the residual stream every iteration:

```python
pos_embed = row_embed(row_idx) + col_embed(col_idx) + box_embed(box_idx)
# ...
h = h_prev + pred_proj(preds) + pos_embed  # every iteration
```

This encodes sudoku's constraint structure directly: cells sharing a row, column, or box get related position vectors.

**Baseline accuracy: 82.8%** (sudoku-extreme, 50K steps, BS=4096, cosine LR)

---

## Early ablations (Kaggle / easy-only era)

### Ablation: No Sudoku Positional Encoding

**File:** `ablation_no_sudoku_pos.py`

**Change:** Replace `row_embed + col_embed + box_embed` with simple learned 81-position embedding.

**Results:** 88.4% cell acc, 409 solved (vs baseline 92.8% acc, 643 solved)

**Finding:** Structured positional encoding helps (+4.4% acc, +234 puzzles). It bakes in sudoku structure (which cells share constraints). However, the gap narrows significantly with better training setups (see abspos below).

### Ablation: Sinusoidal Positional Encoding

**File:** `exp_sinusoidal_pos.py`

**Change:** Replace learned `nn.Embedding(9, d_model)` for row/col/box with fixed sinusoidal encodings: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`.

**Results:** 50.1% acc, 0 solved

**Finding:** This sinusoidal configuration failed. The frequency range may be poorly suited to positions 0-8, but this run does not show that sinusoidal encodings generally fail on small grids.

### RRN Ablation: No Sudoku Positional Encoding

**File:** `rrn_ablation_no_sudoku_pos.py`

**Finding:** In the RRN (Recurrent Relational Network) architecture, structured pos encoding was "not needed" — the message-passing structure already encodes cell relationships.

### Position Embedding Once vs Every Iteration

**File:** `exp_cosine_pos_once.py`

**Hypothesis:** We add pos_embed every iteration. Since h_prev carries forward, is adding it 16x redundant?

**Change:** Add pos_embed only at initialization, not every iteration.

**Results:**

| Metric | Once | Every Iter | Delta |
|--------|------|------------|-------|
| Total | 82.8% | 83.6% | -0.8pp |

**Finding:** Adding pos_embed every iteration helps **+0.8pp**, at little computational cost. We did not directly measure whether the improvement comes from preserving position information in the hidden state.

---

## Sudoku-agnostic alternatives (sudoku-extreme, 2026-02-07)

All runs: d_model=128, n_heads=4, n_layers=4, n_iterations=16, batch_size=4096, cosine LR, 50K steps, reverse curriculum (hard-to-easy).

The goal: replace the sudoku-specific row/col/box embeddings with more general-purpose positional encodings.

### Row+Col only (no box)

**File:** `exp_faster_rowcol.py`

**Change:** Drop `box_embed`, keep `row_embed + col_embed`. Row and column are just 2D grid coordinates — nothing sudoku-specific.

**Results: 82.6%** (-0.2pp vs baseline)

**Finding:** Dropping the box embedding cost only 0.2pp in this comparison.

### 2D RoPE (Rotary Position Embeddings)

**File:** `exp_faster_2drope.py`

**Change:** Replace all additive positional embeddings with 2D Rotary Position Embeddings applied to Q/K in attention. Split head_dim (32) in half: first 16 dims rotated by row index, last 16 by column index. Uses base frequency 10 (not 10000) since positions only range 0-8. Requires custom transformer layer.

**Results: 82.5%** (-0.3pp vs baseline)

**Finding:** Within 0.3pp of baseline. Position information enters through attention rotation at every layer and iteration. The formula supports other grid coordinates, but these experiments tested only 9x9 Sudoku, not accuracy on new grid sizes.

### Learned Absolute Position Embedding

**File:** `exp_faster_abspos.py`

**Change:** Replace row/col/box with single `nn.Embedding(81, d_model)` — one learned vector per cell, zero grid knowledge.

**Results: 81.7%** (-1.1pp vs baseline)

**Finding:** The gap was 1.1pp with this training setup. The early ablation's 4.4pp gap measured cell accuracy on a different dataset, so the two gaps are not directly comparable.

### T5-style Relative Position Bias

**File:** `exp_faster_t5bias.py`

**Change:** No additive position embeddings. Instead, add a learned scalar bias to attention logits based on 1D relative distance (i-j). `nn.Embedding(161, 1)` — 161 = 2*80+1 possible relative distances. Shared across all heads. Passed via `src_mask` to `nn.TransformerEncoder`.

**Results: 73.9%** (-8.9pp vs baseline)

**Finding:** Significantly worse. Flattened distance treats cells (0,8) and (1,0) as neighbors even though they are far apart on the grid, while vertically adjacent cells are 9 apart in the flattened sequence. The float attention mask also made this implementation about 2x slower. All these runs used 50K steps, so slower execution does not mean fewer training updates.

### ALiBi (Attention with Linear Biases)

**File:** `exp_faster_alibi.py`

**Change:** No positional embeddings at all. Fixed per-head linear decay bias on 1D flattened distance: `bias[h,i,j] = -slope_h * |i-j|`. Slopes: geometric series [0.25, 0.0625, 0.0156, 0.0039]. Requires custom transformer layer.

**Results: 18.7%** (-64.1pp vs baseline)

**Finding:** This fixed 1D distance bias performed poorly. Like the T5 variant, it uses flattened distance rather than 2D coordinates; the experiment does not isolate whether that choice explains the full accuracy gap.

---

## Summary

| Experiment | Accuracy | vs Baseline | Position method | Grid knowledge |
|------------|----------|-------------|-----------------|----------------|
| **row+col+box** (baseline) | **82.8%** | — | Learned additive | Sudoku-specific |
| **Row+col only** | **82.6%** | -0.2pp | Learned additive | Grid only |
| **2D RoPE** | **82.5%** | -0.3pp | Q/K rotation | Grid only |
| Abs pos (81) | 81.7% | -1.1pp | Learned additive | None |
| T5 rel bias | 73.9% | -8.9pp | Learned attn bias (1D) | None |
| Sinusoidal | 50.1%* | — | Fixed additive | Sudoku-specific |
| ALiBi | 18.7% | -64.1pp | Fixed attn bias (1D) | None |

*Sinusoidal tested in early era with different training setup, not directly comparable.

## Key Insights

The tested 2D encodings retained almost all baseline accuracy without explicit box information. Learned absolute positions also worked reasonably well. The two flattened-distance biases performed much worse, but these comparisons do not isolate the reason or establish results beyond 9x9 Sudoku.

## Current choice

**2D RoPE** was selected because it came close to row+col+box accuracy while assuming only a 2D grid, not Sudoku's row/column/box rules. See the [README](../README.md) for the current model and training recipe.

## Notes on log locations

- Modal logs are stored in the `sudoku-outputs` volume (download with `modal volume get`).
- Experiment names on volume: `exp_faster_abspos.log`, `exp_faster_t5bias.log`, `exp_faster_rowcol.log`, `exp_faster_2drope.log`, `exp_faster_alibi.log`.
