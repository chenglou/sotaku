# Sudoku Transformer Experiments (Archived)

> **Historical record:** These experiments predate the current model. Baselines, datasets, and evaluation sizes changed during this work; comparisons apply to the settings stated in each section. See the [README](README.md) for current recommendations and research links. Original results are retained below, with speculative explanations separated from observations.

Early experiments use 100k training steps on easiest difficulty puzzles (100k train, 1k test).
Later experiments (curriculum, recurrence) use full 2.7M training set across all difficulties (2.5k test).

**Training infrastructure:** Early experiments ran on RTX 4090. Later experiments (BS=4096, scale_wide, scale_up) ran on Modal H200.

---

## Baseline: Iterative Transformer

**File:** `sudoku.py`

**Architecture:**
- 1 transformer (4 layers, d_model=128, 4 heads)
- 16 iterations with shared weights
- Structured positional encoding (row + col + box embeddings)
- Intermediate supervision (loss at all 16 iterations)
- Input: concat(puzzle, predictions) at each iteration

**Results:** 92.8% acc, 643 solved (peak at step 96k)

---

## Ablation: No Iteration

**File:** `ablation_no_iteration.py`

**Hypothesis:** Is iterative refinement necessary, or can a single forward pass solve Sudoku?

**Change:** Set n_iterations=1 (single forward pass). Intermediate supervision N/A since only one iteration.

**Results:** 64.3% acc, 0 solved

**Finding:** The single-pass version solved no puzzles in this test. This does not establish that single-pass networks cannot solve Sudoku.

---

## Ablation: No Intermediate Supervision

**File:** `ablation_no_intermediate.py`

**Hypothesis:** Does supervising all 16 iterations help, or is final-only loss sufficient?

**Change:** Only compute loss on final iteration output, not all 16.

**Results:** 87.0% acc, 428 solved

**Finding:** Intermediate supervision helps significantly (+5.8% acc, +215 puzzles). It provides gradient signal to early iterations, stabilizing training.

---

## Ablation: No Sudoku Positional Encoding

Moved to [pos_embedding/EXPERIMENTS_POS.md](pos_embedding/EXPERIMENTS_POS.md).

---

## Experiment: Project-then-Add Input

**File:** `exp_proj_add.py`

**Hypothesis:** Is concatenating puzzle and predictions optimal, or would separate projections with addition work better?

**Change:** Instead of single linear on concat(puzzle, preds), use three separate projections added together: proj_digit + proj_pred + empty_embed.

**Results:**
- Original run: 96.0% acc, 817 solved
- Rerun: 91.1% acc, 555 solved (peak 93.78%/719)

**Finding:** Original result was likely a lucky run. Rerun shows high variance and similar performance to baseline. No reliable improvement from separate projections.

---

## Experiment: Project-then-Concat Input

**File:** `exp_proj_concat.py`

**Hypothesis:** Would concatenating separate projections (instead of adding) work better?

**Change:** Project to smaller dimensions that sum to d_model (64+48+16=128), then concatenate.

**Results:** 92.1% acc, 582 solved (peak 92.64%/636)

**Finding:** Similar to project-then-add rerun. No clear advantage over simple concat baseline. Both fancy projection schemes show high training variance without reliable gains.

---

## Experiment: 2 Transformers × 2 Layers (Middle1)

**File:** `exp_middle1.py`

**Hypothesis:** Would specialized early/late phase transformers help? Maybe iterations 1-8 need different processing than 9-16.

**Change:**
- 2 separate transformers with 2 layers each (vs 1 transformer with 4 layers)
- T1 handles iterations 1-8, T2 handles iterations 9-16
- Same ~800k total params, but 32 layer passes (vs 64)

**Results:** 84.2% acc, 162 solved

**Finding:** Significantly worse than baseline. This changed both weight sharing and total computation, so it does not isolate the effect of specialized early and late weights.

---

## Experiment: 4 Transformers × 1 Layer (Middle2)

**File:** `exp_middle2.py`

**Hypothesis:** What if we specialize even more - 4 different transformers for 4 phases?

**Change:**
- 4 separate transformers with 1 layer each
- T1: iters 1-4, T2: iters 5-8, T3: iters 9-12, T4: iters 13-16
- Same ~800k total params, but 16 layer passes (vs 64)

**Results:** 72.7% acc, 0 solved

**Finding:** No puzzles solved. As above, the change reduced total computation as well as changing weight sharing.

---

## Experiment: Unrolled (16 Separate Transformers)

**File:** `exp_unrolled.py`

**Hypothesis:** Does weight sharing help or hurt? If we use 16 separate 4-layer transformers (one per iteration), we get same FLOPs but 16x more params. This tests pure weight sharing effect.

**Change:**
- 16 separate transformers with 4 layers each (vs 1 transformer used 16 times)
- Each iteration uses a different transformer
- Same 64 layer passes (same FLOPs), but ~12.8M params (vs ~800k)

**Results:** 90.1% acc, 581 solved (peak ~566)

**Finding:** Weight sharing performed better in this comparison. Despite 16x more parameters, the unrolled model had:
- Lower accuracy (90.1% vs 92.8%)
- Fewer puzzles solved (581 vs 643)
- Slower training throughout

Weight sharing may act as regularization, but this early comparison does not establish the mechanism. See the later [three-seed weight-sharing study](looping/weight_tying/RESULTS.md) for controlled comparisons on both hard and easier puzzles.

---

## Experiment: Sinusoidal Positional Encoding

Moved to [pos_embedding/EXPERIMENTS_POS.md](pos_embedding/EXPERIMENTS_POS.md).

---

## Experiment: Batch Size Scaling

**Files:** `sudoku.py` (BS=128), `sudoku_bs256.py`, `sudoku_bs512.py`

**Hypothesis:** Larger batch sizes process more samples in same wall-clock time. Does this improve results?

**Change:** Scale batch size while keeping 100k steps constant. This means:
- BS=128: 12.8M samples
- BS=256: 25.6M samples (2x data, ~1.1x wall time)
- BS=512: 51.2M samples (4x data, ~2.2x wall time)

Also added bf16 mixed precision + TF32 for ~2.4x speedup.

**Results:**

| Batch Size | Final Acc | Final Solved | Peak Solved | Samples | Time |
|------------|-----------|--------------|-------------|---------|------|
| 128 | 92.2% | 537 | 679 | 12.8M | ~2.3h |
| 256 | 95.8% | 833 | **897** | 25.6M | ~2.5h |
| 512 | 94.6% | 672 | 866 | 51.2M | ~5h |

**Finding:** BS=256 is the sweet spot:
- Best peak (897 solved) and best final (833 solved)
- Only ~10% slower than BS=128 for 2x the samples
- BS=512 shows diminishing returns - more data but high variance and lower final results
- Larger batches may benefit from LR scaling (not tested)

---

## Experiment: SAM (Sharpness-Aware Minimization)

**Files:** `exp_sam.py` (BS=512), `exp_sam_bs256.py` (BS=256)

**Hypothesis:** Large batch training finds sharp minima that generalize poorly (the "generalization gap"). SAM explicitly seeks flat minima by optimizing for worst-case loss in a weight neighborhood. Can SAM close the gap for large batches?

**Background:** SAM computes gradients at perturbed weights (`w + rho * g/||g||`) and uses those gradients for the update. It is designed to reduce loss sensitivity to nearby weight perturbations. The experiments below measure accuracy, not whether a particular minimum explains the result.

**Results:**

| Config | Peak Solved | Final Solved | Final Acc | Time |
|--------|-------------|--------------|-----------|------|
| Vanilla BS=256 | 897 | 833 | 95.8% | ~2.5h |
| Vanilla BS=512 | 866 | 672 | 94.6% | ~5h |
| **SAM BS=256** | **958** | 930 | 98.1% | ~5.1h |
| **SAM BS=512** | **959** | 948 | 98.6% | ~6.3h |

**Finding:** SAM dramatically improves results:
- SAM BS=512 peak: 959 vs vanilla's 866 (+93 puzzles!)
- SAM BS=512 final: 948 vs vanilla's 672 (+276 puzzles!)
- SAM closes the generalization gap completely - BS=512 now matches BS=256
- SAM pushes past vanilla's ceiling entirely (959 vs 897 best vanilla)
- Overhead is ~25% for BS=512 (6.3h vs 5h), acceptable for massive gains

**Update (post-cosine):** With cosine LR decay, SAM's benefit drops from +6pp to **+0.4pp** (84.0% with SAM, 83.6% without). See the [cosine experiments](cosine/EXPERIMENTS_COSINE.md#cosine-lr-without-sam) for that speed/accuracy comparison.

---

## Experiment: Mixed Difficulty Training

**File:** `train_mixed.py`

**Hypothesis:** Training only on easy puzzles (difficulty 0.0) doesn't generalize to harder puzzles. Will training on a mix of all difficulties help?

**Setup:**
- Sample 20k puzzles from each difficulty bucket (0-1, 1-2, 2-3, 3-4, 4+)
- Total: 100k training puzzles, uniformly mixed
- Test set: 200 from each bucket (1000 total)
- Uses SAM + BS=512 (best config)

**Zero-shot baseline** (model trained on easy only):

| Difficulty | Solved | Cell Acc |
|------------|--------|----------|
| 0.0 (easy) | 959/1000 (95.9%) | 99.0% |
| 1.x | 653/1000 (65.3%) | 90.1% |
| 2.x | 476/1000 (47.6%) | 83.5% |
| 3.x | 279/1000 (27.9%) | 76.2% |
| 4.x | 193/1000 (19.3%) | 73.0% |
| 5.x+ | 125/1000 (12.5%) | 70.4% |

**Mixed training results:**

| Difficulty | Solved | Cell Acc | vs Easy-only |
|------------|--------|----------|--------------|
| 0.0 (easy) | 960/1000 (96.0%) | 98.6% | +1 |
| 1.x | 953/1000 (95.3%) | 98.4% | **+300** |
| 2.x | 904/1000 (90.4%) | 96.6% | **+428** |
| 3.x | 851/1000 (85.1%) | 94.7% | **+572** |
| 4.x | 815/1000 (81.5%) | 93.3% | **+622** |
| 5.x+ | 775/1000 (77.5%) | 91.6% | **+650** |

**Finding:** Mixed training dramatically improves generalization:
- Easy puzzle performance unchanged (~96%)
- Hardest puzzles: 12.5% → 77.5% (+650 puzzles!)
- Cell accuracy stays high across all difficulties (91-99%)
- No curriculum learning needed - uniform mixing works great

---

## Experiment: Hard-Only Training

**File:** `train_hard.py`

**Hypothesis:** If the model learns to solve hard puzzles, easy ones should come "for free" - hard reasoning subsumes easy reasoning.

**Setup:**
- Train only on puzzles with difficulty >= 3.0 (~320k available)
- 100k training steps, SAM + BS=512
- Test on all difficulty levels

**Results:**

| Difficulty | Hard-Only | Mixed | Delta |
|------------|-----------|-------|-------|
| 0.0 (easy) | 913/1000 (91.3%) | 960/1000 (96.0%) | **-47** |
| 1.x | 702/1000 (70.2%) | 953/1000 (95.3%) | **-251** |
| 2.x | 570/1000 (57.0%) | 904/1000 (90.4%) | **-334** |
| 3.x | 879/1000 (87.9%) | 851/1000 (85.1%) | +28 |
| 4.x | 836/1000 (83.6%) | 815/1000 (81.5%) | +21 |
| 5.x+ | 800/1000 (80.0%) | 775/1000 (77.5%) | +25 |

**Finding:** The "hard subsumes easy" hypothesis is **FALSE**:

- Hard-only is better on hard puzzles (3.x+): +21-28 puzzles per bucket
- Hard-only is **much worse** on easy puzzles (0.x-2.x): -47 to -334 puzzles per bucket
- The skills don't transfer bidirectionally

The comparison shows that training only on hard puzzles did not generalize as well to easy puzzles. It does not identify the reasoning strategies learned by either model. Mixed training performed better across the full tested range.

---

## Summary Table

| Experiment | Test Set | Solved | Key Finding |
|------------|----------|--------|-------------|
| Baseline (easy only) | 1k easy | 643 | - |
| No iteration | 1k easy | 0 | Iteration critical |
| No intermediate | 1k easy | 428 | Intermediate helps |
| No sudoku pos | 1k easy | 409 | See [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) |
| Project-add | 1k easy | 555 | No reliable gain |
| Project-concat | 1k easy | 582 | No reliable gain |
| Middle1 (2×2) | 1k easy | 162 | Depth > specialization |
| Middle2 (4×1) | 1k easy | 0 | 1 layer insufficient |
| Unrolled (16×4) | 1k easy | 581 | Weight sharing helps |
| Sinusoidal pos | 1k easy | 0 | See [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) |
| BS=256 + bf16 | 1k easy | 833 | Larger batch helps |
| SAM + BS=512 | 1k easy | 948 | SAM closes gen gap |
| Mixed training | 2.5k mixed | 1930 | Mixed > easy-only |
| Curriculum (easy→hard) | 2.5k mixed | 1790 | Curriculum hurts! |
| **Reverse curriculum** | 2.5k mixed | **1994** | Hard→easy wins |
| **Recurrence (h_prev)** | 2.5k mixed | **2265** | **+13.6% over baseline** |
| Recurrence no preds | 2.5k mixed | 2238 | Preds still helps |
| No x after init | 2.5k mixed | 2248 | Removing x costs only -0.7% |
| Norm pred (TRM-style) | 2.5k mixed | 2257 | RMS norm works, but no gain |
| Eval on sudoku-extreme | 423k extreme | 32.9% | vs nano-trm 87.4% |
| Train on sudoku-extreme | 22k extreme | 63.4% | Domain match +32pp |
| **MLP-Mixer** | 25k extreme | **71.9%** | Same as Transformer |
| Scale DOWN (100K params) | 25k extreme | 8.3% | Too small, fails |
| Scale UP (5M params) | 25k extreme | 69.7% | More params ≠ better |
| **BS=4096** | 25k extreme | **76.3%** | Batch scaling most efficient |
| TRM Nested (4.5M) | 25k extreme | 60.3% | TRM architecture hurts! |
| **LR Warmup** | 25k extreme | **78.5%** | +2.2pp from 2K-step warmup |
| Fixed Random Init | 25k extreme | 77.5% | -1.0pp vs warmup, doesn't help |
| Carry Across Batches | 25k extreme | diverged | Training explodes, doesn't work |
| EMA (decay=0.999) | 25k extreme | 77.5% | -1.0pp vs warmup, doesn't help |
| Cosine LR Decay | 25k extreme | 84.0% | +5.5pp from cosine decay |
| Cosine + Mixed | 25k extreme | 83.8% | Mixed nearly matches reverse with cosine |
| Cosine + Regular | 25k extreme | 80.6% | Easy→hard still hurts (-3.4pp) |
| **Cosine - SAM** | 25k extreme | **83.6%** | **Recommended: 2x faster, -0.4pp** |
| Cosine pos_once | 25k extreme | 82.8% | See [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) |
| **Faster 2D RoPE** | 25k extreme | **82.5%** | New sudoku-agnostic baseline |
| Faster 2D RoPE + 32 test iters | 25k extreme | 88.4% | Free +5.9pp from more test iters |
| **Faster 2D RoPE + confidence stop** | 25k extreme | **91.5%** | **Per-puzzle adaptive stopping** |
| Faster 2D RoPE + oscillation stop | 25k extreme | 91.1% | Causal/online-usable variant |
| Q-head (16 iters) | 25k extreme | 79.4% | Q-loss hurts main task (-3.1pp) |
| Q-head (32 iters) | 25k extreme | 78.4% | Q-loss + halved BS hurts more |

---

## Key Insights

The experiment-specific findings are recorded beside their results rather than repeated here. Later summaries cover [positional encodings](pos_embedding/EXPERIMENTS_POS.md), [learning-rate schedules](cosine/EXPERIMENTS_COSINE.md), [iteration counts and stopping](iters/EXPERIMENTS_ITERS.md), and [weight sharing](looping/weight_tying/RESULTS.md).

---

## Experiment: Curriculum Learning

**Files:** `train_curriculum.py`, `train_curriculum_reverse.py`, `train_mixed.py`

**Hypothesis:** Does the order of difficulty exposure matter? Traditional curriculum learning (easy→hard) is widely used, but maybe for iterative reasoning tasks, starting with hard problems builds better foundations.

**Setup:**
- ~2.7M training puzzles (all kept on CPU, batch moved to GPU per step)
- 100k steps, SAM + BS=512
- Test: 500 puzzles per bucket × 5 = 2500 total
- Train/test split: `iloc[:-500]` for train, `tail(500)` for test (verified 0 overlap)

**Phase schedules:**

| Steps | Curriculum (Easy→Hard) | Reverse (Hard→Easy) |
|-------|------------------------|---------------------|
| 0-20k | 0.0-1.0 only | 3.0+ only |
| 20-40k | 0.0-2.0 | 2.0+ |
| 40-60k | 0.0-3.0 | 1.0+ |
| 60-80k | 0.0-4.0 | All |
| 80-100k | All | All |

Mixed training uses all difficulties from step 0.

**Results:**

| Method | 0.0 | 1.x | 2.x | 3.x | 4.x+ | Total |
|--------|-----|-----|-----|-----|------|-------|
| Curriculum | 98.8% | 84.2% | 69.6% | 60.0% | 45.4% | **1790 (71.6%)** |
| Mixed | 98.4% | 87.0% | 75.0% | 67.8% | 57.8% | **1930 (77.2%)** |
| **Reverse** | 99.6% | 90.4% | 79.0% | 67.8% | 62.0% | **1994 (79.8%)** |

**Key findings:**

1. **Reverse curriculum wins** (+204 over curriculum, +64 over mixed)
2. **Traditional curriculum hurts** - actually worse than mixed training
3. **Hard puzzles benefit most**: reverse achieves 62% vs curriculum's 45.4% (37% relative improvement)
4. **No trade-off on easy**: Reverse scores highest on easy puzzles too (99.6%)

**Conclusion:** Hard-to-easy training performed best in these runs. We did not measure whether the models learned different reasoning strategies, so the results do not establish a general rule for iterative reasoning tasks.

---

## Inconclusive: Scaling Experiments (BS confounded)

**Files:** `exp_scale_model.py`, `exp_scale_iter.py`

**Note:** These experiments are inconclusive because they required reducing batch size from 512 to 256 due to memory constraints. The BS change is a significant confounder since BS=512+SAM was our optimal training config.

### Scale Model (d=192, L=6)

**Hypothesis:** Larger model capacity might improve performance, especially on hard puzzles.

**Change:**
- d_model: 128 → 192
- n_layers: 4 → 6
- n_heads: 4 → 6
- d_ff: 512 → 768
- ~3x parameters (~2.4M vs ~800k)
- BS: 512 → 256 (memory constraint)

**Results:** 1888/2500 (75.5%) vs baseline 1994/2500 (79.8%)

| Difficulty | Scale Model | Baseline | Delta |
|------------|-------------|----------|-------|
| 0.0 | 98.6% | 99.6% | -1.0% |
| 1.x | 86.8% | 90.4% | -3.6% |
| 2.x | 73.4% | 79.0% | -5.6% |
| 3.x | 61.8% | 67.8% | -6.0% |
| 4.x+ | 57.0% | 62.0% | -5.0% |

**Observation:** Performed worse than baseline, but unclear if due to smaller BS or model size. Peak was 1982/2500 at step 70k.

### Scale Iterations (32)

**Hypothesis:** More iterations = more "thinking time" for constraint propagation, should help hard puzzles.

**Change:**
- n_iterations: 16 → 32
- BS: 512 → 256 (memory constraint)

**Results:** 1559/2500 (62.4%) vs baseline 1994/2500 (79.8%)

| Difficulty | 32 Iter | Baseline | Delta |
|------------|---------|----------|-------|
| 0.0 | 89.2% | 99.6% | -10.4% |
| 1.x | 73.0% | 90.4% | -17.4% |
| 2.x | 57.2% | 79.0% | -21.8% |
| 3.x | 50.4% | 67.8% | -17.4% |
| 4.x+ | 42.0% | 62.0% | -20.0% |

**Observation:** Catastrophic failure. Training was extremely unstable:
- Loss spikes to 1.5+ throughout training
- Complete collapse at step 90k (solved 1/2500 puzzles)
- Never recovered to competitive performance

**Why 32 iterations might have failed:**
1. Intermediate supervision on all 32 iterations may cause gradient issues
2. Shared weights through 32 iterations may be unstable
3. 4-layer transformer may not have enough capacity per iteration
4. Smaller BS (256 vs 512) may not provide enough gradient stability

**To properly test these hypotheses, need to:**
1. Run baseline at BS=256 for fair comparison
2. Try 32 iterations with final-only loss (no intermediate supervision)
3. Try larger model + more iterations together

---

## Experiment: Hidden State Recurrence

**Files:** `exp_recur_add.py`, `exp_recur_concat.py`, `exp_recur_gated.py`, `exp_recur_mem.py`

**Hypothesis:** The baseline only passes predictions (9-dim softmax) between iterations - the hidden state h is recomputed from scratch each time. What if we pass the full hidden state (128-dim) forward, giving the model a "scratchpad" for working memory?

**Background:** Analysis of failures ([archived analyze_failures.py](https://github.com/chenglou/sotaku/blob/v2.0.0/analyze_failures.py)) showed that on failed puzzles:
- Model peaks at iteration 4, then gets *worse* (oscillates without converging)
- 99.4% of failures still changing at final iteration (not converged)
- Successful puzzles show steady improvement across iterations

This suggests the model lacks persistent memory to accumulate reasoning across iterations.

**Baseline architecture:** Each iteration computes h from scratch via `transformer(input_proj(concat(x, preds)))`. Only the 9-dim preds carries forward; the 128-dim hidden state h is discarded.

**Four recurrence variants tested:**

| Variant | Change | Extra Params |
|---------|--------|--------------|
| **recur_add** | `h = ... + h_prev` | None |
| **recur_concat** | `input_proj(concat(x, preds, h_prev))` | +16k (input proj larger) |
| **recur_gated** | `h = gate * h_prev + (1-gate) * h_new` | +33k (gate projection) |
| **recur_mem** | Separate 64-dim memory bank, accumulated | +16k (mem projections) |

**Results:**

| Model | Total | 0.0 | 1.x | 2.x | 3.x | 4.x+ |
|-------|-------|-----|-----|-----|-----|------|
| Baseline | 1994 | 498 | 450 | 395 | 338 | 305 |
| **recur_add** | **2265** | 498 | 483 | 461 | 424 | **399** |
| recur_concat | 2206 | 500 | 483 | 447 | 403 | 373 |
| recur_gated | 2141 | 499 | 464 | 441 | 389 | 348 |
| recur_mem | 2254 | 500 | 487 | 462 | 423 | 382 |

**Key findings:**

1. **Simplest approach wins**: Just adding `h_prev` (no extra params!) gives best results
2. **Huge improvement on hard puzzles**: 305 → 399 (+94 puzzles, **+31%**)
3. **Overall improvement**: 1994 → 2265 (+271 puzzles, **+13.6%**)
4. **Gated recurrence underperforms**: GRU-style gating may be too complex, learning to gate away useful information
5. **All recurrence methods beat baseline**: Even the worst (gated, 2141) significantly outperforms baseline (1994)

**Interpretation:** Passing the 128-dimensional hidden state lets later iterations use information beyond the nine digit probabilities. These experiments did not identify what that information represents or why addition outperformed gating.

---

## Ablation: Recurrence Without Predictions

**Files:** `exp_recur_add_nopred.py`, `exp_recur_concat_nopred.py`, `exp_recur_mem_nopred.py`

**Hypothesis:** If h_prev contains all the information (including what led to predictions), is the explicit `preds` input redundant? Since `preds = softmax(output_head(h))`, passing h_prev should be strictly more informative.

**Change:** Remove preds from input, rely solely on h_prev for iteration state.

**Results:**

| Model | With Preds | No Preds | Delta |
|-------|------------|----------|-------|
| add | **2265** | 2238 | -27 |
| concat | 2206 | 2204 | -2 |
| mem | **2254** | 2192 | -62 |

**Finding:** Removing preds **hurts** across the board, especially for the memory variant (-62 puzzles).

**Conclusion:** Keeping both preds and h_prev performed best here. The experiment does not establish why supplying the predictions explicitly helps.

---

## Experiment: No X After Init (TRM-style Input)

**File:** `exp_no_x_after_init.py`

**Hypothesis:** TRM encodes the puzzle x once at initialization and never re-feeds it. Can we do the same? This tests whether the model needs fresh x input at every iteration or if the hidden state h_prev captures all necessary puzzle info.

**Change:** Encode x once at init into h_prev. In the loop, only use h_prev + pred_proj(preds) + pos_embed — no x re-fed.

**Results:** 2248/2500 (89.9%) vs baseline 2265/2500 (90.6%) = -17 puzzles (-0.7%)

**Finding:** Encoding x only at initialization cost 0.7pp in this comparison. The hidden state retained enough puzzle information for similar accuracy, though this does not prove it preserved every relevant detail.

---

## Experiment: Normalized Prediction Residuals (TRM-style)

**File:** `exp_norm_pred.py`

**Hypothesis:** TRM updates its prediction state with residual connections + RMS normalization. Our earlier `exp_separate_h_pred` failed because logits accumulated unboundedly. Can we make predictions update separately with proper normalization?

**Change:** h updates independently (no preds input). Separate prediction state with residual + RMS norm: `pred_state = rms_norm(pred_state + MLP(concat(h, pred_state)))`. Final logits = output_head(h) + pred_state.

**Results:** 2257/2500 (90.3%) vs baseline 2265/2500 (90.6%) = -8 puzzles (-0.3%)

| Difficulty | norm_pred | recur_add (baseline) |
|------------|-----------|----------------------|
| 0.x | 500/500 | 498/500 |
| 1.x | 483/500 | 483/500 |
| 2.x | 461/500 | 461/500 |
| 3.x | 420/500 | 424/500 |
| 4.x+ | 393/500 | 399/500 |

**Finding:** RMS normalization successfully prevents the explosion that killed `exp_separate_h_pred`. However, the added complexity (~70K extra params for pred_update MLP) doesn't improve results. Simple `h_prev` addition remains the best approach - more complex prediction update mechanisms don't help.

---

## Benchmark: Sudoku-Extreme

**File:** [archived eval_extreme.py](https://github.com/chenglou/sotaku/blob/v2.0.0/eval_extreme.py)

**Dataset:** [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) - 423k test puzzles rated by backtrack count (higher = harder).

**Comparison with [nano-trm](https://github.com/olivkoch/nano-trm)** (87.4% on this benchmark):

| Rating | Baseline | recur_add | TRM |
|--------|----------|-----------|-----|
| 0 (trivial) | 98.3% | 99.8% | - |
| 1 | 78.7% | 93.4% | - |
| 2 | 68.0% | 84.2% | - |
| 3-5 | 27.9% | 40.1% | - |
| 6-10 | 3.3% | 12.1% | - |
| 11-20 | 1.8% | 11.3% | - |
| 21-50 | 1.4% | 10.4% | - |
| 51+ | 0.8% | 6.6% | - |
| **Total** | 24.4% | **32.9%** | **87.4%** |

**Analysis:**

Recurrence helped (+8.5pp overall), but a gap remained to nano-trm. Later architecture and size experiments below did not close that gap; the [cosine experiments](cosine/EXPERIMENTS_COSINE.md) found a larger improvement from the learning-rate schedule.

---

## Experiment: Training on Sudoku-Extreme

**File:** `exp_extreme_curriculum.py`

**Hypothesis:** Our models trained on Kaggle data only achieve 31.7% on sudoku-extreme. Does training directly on sudoku-extreme close the gap to TRM?

**Setup:**
- 400K training puzzles from sudoku-extreme (test split, before we knew train split existed)
- Same architecture as `no_x_after_init` (encode x once)
- Reverse curriculum based on rating: Phase 1 (rating 21+) → Phase 2 (6+) → Phase 3 (1+) → Phase 4 (all)
- 100K steps, SAM + BS=512

**Results:**

| Trained on | Kaggle test (2.5K) | sudoku-extreme test (22K) |
|------------|-------------------|---------------------------|
| Kaggle 2.7M | **89.9%** | 31.7% |
| sudoku-extreme 400K | 81.3% | **63.4%** |
| nano-trm (reference) | - | **87.4%** |

**By difficulty on sudoku-extreme:**

| Rating | Kaggle-trained | Extreme-trained |
|--------|----------------|-----------------|
| 0 (trivial) | 99.8% | 98.7% |
| 1-2 | 88.9% | 79.8% |
| 3-10 | 62.0% | 51.5% |
| 11-50 | 21.5% | 53.4% |
| 51+ | 6.6% | 62.8% |

**Key findings:**

1. **Domain match matters hugely**: +32pp on sudoku-extreme (31.7% → 63.4%) with 7x less data
2. **Not universally better data**: -8.6pp on Kaggle (89.9% → 81.3%), models specialize to their domain
3. **Still far from TRM**: 63.4% vs 87.4% despite training on the same domain. The cause of the remaining gap was not isolated.
4. **Hard puzzles benefit most**: Rating 51+ jumps from 6.6% to 62.8% (+56pp!)

**Stabilization:** Similar to Kaggle experiments - rapid gains until 50K, plateau with variance 50-100K. 70K steps sufficient.

---

## New Baseline: 2.7M Sudoku-Extreme

**File:** `exp_extreme_baseline.py`

**Hypothesis:** Our previous sudoku-extreme experiment used only 400K puzzles. What if we match the Kaggle data quantity (2.7M)?

**Setup:**
- 2.7M training puzzles from sudoku-extreme train split (matching Kaggle quantity)
- Same architecture as `no_x_after_init` (encode x once)
- Reverse curriculum based on rating: Phase 1 (rating 21+) → Phase 2 (6+) → Phase 3 (1+) → Phase 4 (all)
- 70K steps (not 100K - training stabilizes by then), SAM + BS=512

**Results:**

| Trained on | Data | Kaggle test (2.5K) | sudoku-extreme test |
|------------|------|-------------------|---------------------|
| Kaggle | 2.7M | **89.9%** | 31.7% |
| sudoku-extreme | 400K | 81.3% | 63.4% |
| **sudoku-extreme** | **2.7M** | 83.3% | **71.4%** |
| nano-trm (reference) | 1K | - | 87.4% |

**By difficulty on sudoku-extreme test:**

| Rating | 400K trained | 2.7M trained |
|--------|--------------|--------------|
| 0 (trivial) | 4812/5000 (96.2%) | 4935/5000 (98.7%) |
| 1-2 | 3888/5000 (77.8%) | 4154/5000 (83.1%) |
| 3-10 | 2513/5000 (50.3%) | 2999/5000 (60.0%) |
| 11-50 | 2617/5000 (52.3%) | 3283/5000 (65.7%) |
| 51+ | 2874/5000 (57.5%) | 3563/5000 (71.3%) |

**Key findings:**

1. **More data helps**: +8pp on sudoku-extreme (63.4% → 71.4%) by increasing from 400K to 2.7M
2. **Cross-domain also improves**: +2pp on Kaggle (81.3% → 83.3%) despite never seeing Kaggle puzzles
3. **Still far from nano-trm**: 71.4% vs 87.4%; later architecture changes did not close the gap.
4. **Hard puzzles benefit most**: Rating 51+ jumps from 57.5% to 71.3% (+14pp)

**This is now the new baseline** for sudoku-extreme experiments: 2.7M data, 70K steps, no_x_after_init architecture

---

## Experiment: MLP-Mixer (Architecture Swap)

**File:** `exp_mlp_mixer.py`

**Hypothesis:** TRM uses MLP-Mixer instead of Transformer attention. Is their 87.4% vs our 71.4% due to the architecture difference?

**Change:** Replace TransformerEncoder with MLP-Mixer layers (token mixing MLP across 81 positions + channel mixing MLP). Everything else identical: same looping, h_prev recurrence, pos embeddings, training setup.

**Parameters:** 870K (MLP-Mixer) vs 800K (Transformer) - similar

**Results:**

| Model | Total | Rating 0 | 1-2 | 3-10 | 11-50 | 51+ |
|-------|-------|----------|-----|------|-------|-----|
| **MLP-Mixer** | **71.9%** | 99.1% | 86.6% | 53.3% | 56.9% | 63.4% |
| Transformer | 71.4% | 98.7% | 83.1% | 60.0% | 65.7% | 71.3% |
| nano-trm | 87.4% | - | - | - | - | - |

**Finding:** MLP-Mixer achieves **71.9%** vs Transformer's **71.4%** - essentially identical (+0.5pp).

Interesting pattern: MLP-Mixer is *worse* on hard puzzles (51+: 63.4% vs 71.3%) but *better* on easy (1-2: 86.6% vs 83.1%). The architectures trade off differently across difficulties but converge to the same overall accuracy.

**Conclusion:** This MLP-Mixer substitution did not close the gap to nano-trm. It does not rule out other architectural differences or interactions with the training recipe.

---

## Experiment: Scale DOWN

**File:** `exp_scale_down.py`

**Hypothesis:** How small can we go? Testing if a smaller model can still solve Sudoku.

**Change:**
- d_model: 128 → 64
- n_layers: 4 → 2
- n_heads: 4 → 2
- d_ff: 512 → 256
- ~100K params (vs baseline ~800K)

**Results:**

| Rating | Solved |
|--------|--------|
| 0 (easy) | 32.2% |
| 1-2 | 1.4% |
| 3-10 | 1.8% |
| 11-50 | 4.2% |
| 51+ | 2.1% |
| **Total** | **8.3%** |

**Finding:** Catastrophic failure. The small model can barely solve easy puzzles (32%) and essentially nothing harder.

Both width and depth changed, so the result does not isolate which reduction caused the loss of accuracy or establish a minimum viable model size.

---

## Experiment: Scale UP

**File:** `exp_scale_up.py`

**Hypothesis:** TRM has 5M params vs our 800K. Is model size the bottleneck? Let's scale up to match TRM.

**Change:**
- d_model: 128 → 256
- n_layers: 4 → 8
- n_heads: 4 → 8
- d_ff: 512 → 1024
- ~5M params (matching TRM)
- Used gradient accumulation (micro_batch=128 × 4 = effective BS=512) due to GPU memory

**Results:**

| Rating | Scale UP (5M) | Baseline (800K) |
|--------|---------------|-----------------|
| 0 (easy) | 98.2% | 98.7% |
| 1-2 | 80.8% | 83.1% |
| 3-10 | 50.8% | 60.0% |
| 11-50 | 55.5% | 65.7% |
| 51+ | 63.0% | 71.3% |
| **Total** | **69.7%** | **71.4%** |

**Reference:** TRM (5M params): 87.4%

**Finding:** This larger configuration performed worse (-1.7pp), despite learning faster early on (51% vs 12% at step 5K). It did not match nano-trm's 87.4%, but one unsuccessful scaling run does not rule out model size as a factor. The next experiment also changed the result by removing gradient accumulation.

---

## Experiment: Scale UP with True Batch Size

**File:** `exp_scale_up_big_gpu.py`

**Hypothesis:** The previous Scale UP used gradient accumulation (micro_batch=128 × 4). Rerun on H200 GPU with a single BS=512 batch per update to compare the implementations.

**Setup:**
- Same architecture as Scale UP: d_model=256, n_layers=8, n_heads=8, d_ff=1024 (~6.3M params)
- True BS=512 (no gradient accumulation)
- 70K steps on sudoku-extreme 2.7M

**Results:**

| Rating | True BS (6.3M) | Grad Accum (5M) | Baseline (800K) |
|--------|----------------|-----------------|-----------------|
| 0 | 98.7% | 98.2% | 98.7% |
| 1-2 | 85.6% | 80.8% | 83.1% |
| 3-10 | 57.2% | 50.8% | 60.0% |
| 11-50 | 58.5% | 55.5% | 65.7% |
| 51+ | 67.3% | 63.0% | 71.3% |
| **Total** | **73.5%** | **69.7%** | **71.4%** |

**Finding:** This run improved by 3.8pp over the gradient-accumulation version and beat baseline by 2.1pp. The comparison does not establish the source of the difference.

---

## Experiment: Scale WIDE

**File:** `exp_scale_wide.py`

**Hypothesis:** Instead of scaling depth (more layers), what if we scale width (larger d_model)?

**Setup:**
- d_model: 128 → 512
- n_layers: 4 (unchanged)
- d_ff: 512 → 2048
- ~3.2M params
- 70K steps, BS=512

**Results:**

| Rating | Scale WIDE (3.2M) | Scale UP (6.3M) | Baseline (800K) |
|--------|-------------------|-----------------|-----------------|
| 0 | 99.4% | 98.7% | 98.7% |
| 1-2 | 85.6% | 85.6% | 83.1% |
| 3-10 | 58.0% | 57.2% | 60.0% |
| 11-50 | 61.4% | 58.5% | 65.7% |
| 51+ | 69.5% | 67.3% | 71.3% |
| **Total** | **74.8%** | **73.5%** | **71.4%** |

**Finding:** Width scaling (74.8%) beats depth scaling (73.5%) with fewer params (3.2M vs 6.3M). Width is more parameter-efficient than depth for this task.

---

## Experiment: Batch Size Scaling on Sudoku-Extreme

**Files:** `exp_scale_batch.py` (BS=2048), `exp_scale_batch_4k.py` (BS=4096)

**Hypothesis:** Earlier experiments showed BS=256-512 was optimal for small datasets. With 2.7M training puzzles, can we push batch size higher?

**Setup:**
- Same baseline architecture: d_model=128, n_layers=4 (~800K params)
- Scale BS: 512 → 2048 → 4096
- Keep LR=1e-3 for BS=2048, LR=1.5e-3 for BS=4096
- 70K steps on sudoku-extreme 2.7M

**Results:**

| Batch Size | Params | Total | Rating 51+ |
|------------|--------|-------|------------|
| 512 (baseline) | 800K | 71.4% | 71.3% |
| 2048 | 800K | 73.7% | 67.8% |
| 4096 | 800K | **76.3%** | **72.3%** |
| nano-trm (ref) | 5M | 87.4% | - |

**Learning curve (BS=4096):**

| Step | Test Acc |
|------|----------|
| 5K | 54.9% |
| 10K | 64.1% |
| 20K | 68.5% |
| 30K | 73.1% |
| 40K | 73.8% |
| 50K | 75.4% |
| 70K | **76.3%** |

**Finding:** Batch size scaling is the most efficient lever we've found:
- BS=4096 (76.3%) beats 8x more params (73.5%) with zero extra cost per sample
- +4.9pp over baseline just from larger batches
- Gap to TRM reduced from 16pp to 11pp

---

## Experiment: Curriculum Scaling at Large Batch Size

**Files:** `exp_scale_batch_4k_v2.py` (reverse), `exp_scale_batch_4k_curriculum.py` (regular)

**Hypothesis:** Our reverse curriculum was designed for BS=512. With BS=4096, each step sees 8x more data. Should phase boundaries scale?

**Setup:**
- BS=4096 for both experiments
- 10K steps (same total data as BS=512 @ ~80K steps)
- Scaled phases: 0-2K, 2K-4K, 4K-6K, 6K-10K

| Curriculum | Phase Order | Steps | Accuracy |
|------------|-------------|-------|----------|
| Reverse (scaled) | hard→easy | 10K | **70.5%** |
| Regular (scaled) | easy→hard | 10K | 67.1% |

**Same-data efficiency comparison:**

At 41M samples (10K steps @ BS=4096):
- Scaled reverse curriculum: **70.5%**
- Unscaled reverse curriculum: **64.1%**

Scaled phases are **+6.4% more efficient** at the same data budget.

**Learning curves:**

Reverse: 0% → 3% → 36% → 63% → 68.5% → 70.5% (keeps improving). Regular: 0% → 31% → 52% → 65% → 68.1% → 67.1% (drops at end).

**Key observations:**
1. Regular curriculum **overfits**: peaks at 68.2% (step 8K), then drops to 67.1%
2. Loss spikes visible at phase transitions (14K, 42K in unscaled runs)
3. Reverse curriculum keeps improving; regular curriculum degrades after step 8K

**Finding:** Reverse curriculum still wins by +3.4pp at large batch size. Scaling phases improves data efficiency, but more training data still wins overall (76.3% with 70K steps vs 70.5% with 10K steps).

---

## Experiment: TRM Architecture (Nested H_cycles/L_cycles)

**File:** `exp_trm_nested.py`

**Hypothesis:** nano-trm achieves 87.4% on sudoku-extreme. They use a specific architecture with nested iteration loops (H_cycles × L_cycles) where outer loops run without gradients. Does adopting their architecture close the gap?

**Setup:**
- TRM-style MLP-T architecture (sequence-wise MLP instead of attention)
- hidden_size=512, L_layers=2 (matching TRM)
- H_cycles=3, L_cycles=6 (first 2 outer loops without gradients)
- ~4.5M params
- Trained on our 2.7M sudoku-extreme data with reverse curriculum
- BS=4096, lr=1e-4, weight_decay=1.0
- 70K steps

**Results:**

| Rating | TRM Nested | Baseline (BS=4096) | nano-trm (ref) |
|--------|------------|-------------------|-----------------|
| 0 (easy) | 95.1% | 98.7% | - |
| 1-2 | 73.7% | 83.1% | - |
| 3-10 | 41.5% | 60.0% | - |
| 11-50 | 44.4% | 65.7% | - |
| 51+ | 46.9% | 71.3% | - |
| **Total** | **60.3%** | **76.3%** | **87.4%** |

**Finding:** TRM's architecture **hurts** performance significantly (-16pp vs baseline).

This comparison changed the architecture, learning rate, and weight decay together; it does not isolate the effect of running initial iterations without gradients. A follow-up intended to reproduce the nano-trm recipe (`exp_trm_exact.py`) also performed poorly (~14% accuracy). Neither result rules out benefits from individual parts of that recipe. Later [training on later iterations](looping/EXPERIMENTS_LOOPING.md) succeeded without adopting the full TRM architecture.

---

## Experiment: LR Warmup

**File:** `exp_warmup.py`

**Hypothesis:** nano-trm uses 2000-step linear LR warmup. Large batch training can have unstable early gradients - does warmup help?

**Setup:**
- Same as BS=4096 baseline
- Added 2000-step linear LR warmup (0 → 1.5e-3)
- 70K steps on H200

**Results:**

| Metric | With Warmup | Baseline (no warmup) |
|--------|-------------|----------------------|
| Rating 0 | 99.6% | 98.7% |
| Rating 1-2 | 90.2% | 83.1% |
| Rating 3-10 | 62.3% | 60.0% |
| Rating 11-50 | 66.2% | 65.7% |
| Rating 51+ | 74.0% | 72.3% |
| **Total** | **78.5%** | **76.3%** |

**Learning curve comparison:**

| Step | With Warmup | Baseline | Delta |
|------|-------------|----------|-------|
| 5K | 57.1% | 54.9% | +2.2pp |
| 10K | 66.4% | 64.1% | +2.3pp |
| 25K | 70.9% | 70.2% | +0.7pp |
| 50K | 76.9% | 75.4% | +1.5pp |
| 70K | **78.5%** | 76.3% | **+2.2pp** |

**Finding:** LR warmup gives a solid **+2.2pp improvement** (76.3% → 78.5%). The benefit is most visible early (helps early training stability) and persists to the end. Gap to nano-trm reduced from 11.1pp to 8.9pp.

**Remaining gap with nano-trm:**
- Baseline (no warmup): 76.3% → 11.1pp gap
- **With warmup: 78.5% → 8.9pp gap**
- nano-trm: 87.4%

Still need to test: EMA, cosine LR decay, carry across batches, no-grad warmup cycles.

---

## Experiment: Fixed Random Hidden State Init

**File:** `exp_fixed_init.py`

**Hypothesis:** nano-trm initializes hidden states (z_H, z_L) as fixed random buffers with std=1.0, not learned parameters. Does using a fixed random init instead of our learned `initial_encoder(x)` help?

**Setup:**
- Based on warmup baseline (78.5%)
- Replaced `self.initial_encoder = nn.Linear(10, d_model)` with `nn.Buffer` initialized with std=1.0
- Added `puzzle_proj(x)` inside the loop to feed puzzle info (since h_init no longer encodes it)
- 70K steps on H200

**Change:** Replaced learned `initial_encoder(x)` with a fixed random buffer (std=1.0), added `puzzle_proj(x)` inside the loop instead.

**Results:**

| Metric | Fixed Init | Warmup Baseline |
|--------|------------|-----------------|
| Rating 0 | 99.6% | 99.6% |
| Rating 1-2 | 88.8% | 90.2% |
| Rating 3-10 | 60.9% | 62.3% |
| Rating 11-50 | 64.6% | 66.2% |
| Rating 51+ | 73.7% | 74.0% |
| **Total** | **77.5%** | **78.5%** |

**Learning curve:**

| Step | Fixed Init | Warmup | Delta |
|------|------------|--------|-------|
| 10K | 65.2% | 66.4% | -1.2pp |
| 50K | 76.6% | 76.9% | -0.3pp |
| 55K | 76.9% | ~77% | ~0pp |
| 70K | **77.5%** | **78.5%** | **-1.0pp** |

**Finding:** Fixed random init **hurts** by 1.0pp (78.5% → 77.5%). Our learned `initial_encoder(x)` that encodes the puzzle into the initial hidden state is better than nano-trm's fixed random buffer approach.

**Why it didn't help:**
- nano-trm has TWO hidden states (z_H for solution, z_L for problem) that interact differently
- Our single h_prev benefits from being puzzle-aware from the start
- The fixed init forces the model to rely entirely on `puzzle_proj(x)` added each iteration, which may be less effective than encoding puzzle info once into h_prev

**Conclusion:** This nano-trm technique does NOT transfer to our architecture. Crossed off the list.

---

## Experiment: Carry Hidden State Across Batches

**File:** `exp_carry.py`

**Hypothesis:** nano-trm keeps the same puzzle in a batch slot until solved, carrying hidden state forward. This lets the model "think longer" on hard puzzles. Does this help our architecture?

**Setup:**
- Based on warmup baseline (78.5%)
- Persistent slot system: each batch slot holds a puzzle until solved, then replaced
- Hidden state (h_prev) and predictions carry forward for unsolved puzzles
- 70K steps on H200

**Results:** Training diverged catastrophically. Loss exploded to millions by step 20K, accuracy stuck at 11% (random), zero puzzles solved.

**Finding:** Carry across batches **completely breaks** our architecture. The hidden state accumulation causes training instability. This technique does NOT transfer from nano-trm.

**Why it failed:**
- nano-trm has different architecture (two-state z_H/z_L system)
- Their "iterations" within a step work differently
- Our single h_prev may accumulate in ways that destabilize gradients
- The carry mechanism may need their specific normalization or gating

**Conclusion:** Crossed off the list. Will test EMA and cosine LR decay independently.

---

## Experiment: EMA (Exponential Moving Average)

**File:** `exp_ema.py`

**Hypothesis:** nano-trm uses EMA with decay=0.999 for evaluation. Does maintaining shadow weights that average over training improve generalization?

**Setup:**
- Based on warmup baseline (78.5%)
- Added EMA class that maintains shadow weights updated after each step: `shadow = decay * shadow + (1-decay) * params`
- Evaluation uses EMA weights, training uses live weights
- decay=0.999 (same as nano-trm)
- 70K steps on H200

**Results:**

| Metric | With EMA | Warmup Baseline |
|--------|----------|-----------------|
| Rating 0 | 99.5% | 99.6% |
| Rating 1-2 | 89.6% | 90.2% |
| Rating 3-10 | 61.6% | 62.3% |
| Rating 11-50 | 64.8% | 66.2% |
| Rating 51+ | 72.0% | 74.0% |
| **Total** | **77.5%** | **78.5%** |

**Finding:** EMA **hurts** by 1.0pp (78.5% → 77.5%). The shadow weights averaged over training are worse than the final live weights.

**Conclusion:** EMA did not help this configuration. The [later cosine retest](cosine/EXPERIMENTS_COSINE.md#ema-retest) found no accuracy difference with or without EMA.

---

Continued in:
- [cosine/EXPERIMENTS_COSINE.md](cosine/EXPERIMENTS_COSINE.md) — Cosine LR experiments
- [iters/EXPERIMENTS_ITERS.md](iters/EXPERIMENTS_ITERS.md) — Iteration scaling experiments, including the historical 98.9% checkpoint
