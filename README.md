# Parameter Golf — A Study in Extreme Language-Model Compression

A from-the-ground-up reimplementation and extension of the parameter-efficiency techniques behind
OpenAI's [**Parameter Golf**](https://github.com/openai/parameter-golf) challenge (the first OpenAI
"Model Craft" challenge): *train the best language model you can that fits in a **16&nbsp;MB artifact**
and trains in **10 minutes on 8×H100 GPUs**, scored by tokenizer-agnostic **bits-per-byte (BPB)** on
the FineWeb dataset.*

I worked from the public repository, dataset, and evaluation
> harness to understand and implement the techniques that define the leaderboard. All training/model code
> here is built on OpenAI's reference `train_gpt_mlx.py`, modified for study (see **Attribution**). Where I
> report numbers, **storage/parameter measurements are my own and reproducible on any machine**; **quality
> (BPB) figures are cited from the public leaderboard, not self-reported**, because a trustworthy quality
> number requires the full 8×H100 run the challenge specifies.

---

## The problem in one paragraph

The 16&nbsp;MB cap covers the *compressed* model weights plus the training code, so the real constraint is a
**source-coding problem**, not a parameter count. BPB measures how many bits, on average, the model needs to
predict each byte of held-out text — and a model that predicts text well *is* a model that compresses it well
(being unsurprised by the next token is literally compression). Lower is better. OpenAI's reference baseline
scores **1.2244 BPB**; the public leaderboard frontier reached roughly **1.08** using the techniques studied
here. The whole game is getting *more effective model into the same 16&nbsp;MB*: every improvement is a trade,
not an addition, because the baseline already fills the budget.

## Techniques implemented and measured

Each technique below is implemented in code and toggled via an environment variable. Storage effects are
deterministic and were measured directly; the precision/quality intuition is grounded in the public board.

| Technique | Mechanism | Measured / studied effect |
|---|---|---|
| **Grouped-query attention** (baseline) | Share key/value projections across query heads | ~14% fewer parameters vs. full multi-head, ~no quality loss |
| **Depth recurrence** | Reuse a few transformer blocks for multiple passes — decouples *processing depth* from *stored weights* | 9 distinct blocks (**17.1M params**) → 3 blocks looped ×3 (**6.0M params**), same 9 effective passes; **~11 MB of weight budget freed** |
| **Low-bit quantization** (configurable) | Map weights to integer buckets (per-row scales), then entropy-code (zlib); `QUANT_BITS` knob | Implemented the precision↔size↔damage tradeoff. Sweet spot mirrors the board: **int5–6** keeps quality; ternary/1-bit lose too much |
| **Factorized embeddings** | Replace the `V×d` table with `V×r` + shared `r×d` (a rank bottleneck — same low-rank math as LoRA) | **62% smaller embedding** at rank 128; payoff scales with vocabulary size |
| **Sliding-window attention** | Restrict each query to a recent window of keys | Attention cost **O(T²) → O(T·W)**; depth buys back long-range reach via multi-hop propagation |
| **Quantization-aware training** *(studied)* | Simulate rounding during training so weights survive compression; gradients via the **straight-through estimator** | Why leaders push to int5–6 without quality collapse |
| **Test-time training** *(studied)* | Per-document low-rank (LoRA) adaptation at eval, under a strict **causal "score-first"** rule (no peeking ahead) | The current leaderboard frontier; ties the LoRA/low-rank idea back to factorized embeddings |

## What this project demonstrates (the understanding)

The artifact of this project is depth of understanding, defensible at a whiteboard:

- **Bits-per-byte** — why next-token prediction *is* compression, and why dividing by raw bytes makes the
  metric tokenizer-agnostic (and why a lossy tokenizer breaks it).
- **The Transformer, end to end** — embedding → residual stream → grouped-query attention (with RoPE
  injecting relative position by rotation) → relu² feed-forward → tied unembedding → cross-entropy.
- **Parameter-efficiency under a hard cap** — grouped-query attention, depth recurrence, factorized
  embeddings, sliding-window attention, and the size-vs-quality frontier each technique moves.
- **Quantization** — mapping floats to integer buckets, the precision/size tradeoff, the two-stage
  quantize-then-entropy-code pipeline, and quantization-aware training.
- **The Muon optimizer** — what it means to *orthogonalize* a gradient (flatten its singular values while
  preserving directions), why that accelerates training, and how the Newton-Schulz iteration approximates a
  GPU-hostile singular value decomposition using only matrix multiplications.
- **Training systems** — mixed-precision (bf16 compute / fp32 master weights), gradient accumulation for a
  large effective batch on limited memory, and a wall-clock-keyed learning-rate schedule that lands at zero
  exactly as the compute budget expires.

## Repository contents

```
train_gpt_mlx.py   # OpenAI's reference MLX trainer, modified to add the techniques above
README.md          # this file
```

The dataset (FineWeb shards) and the full reference repo are not committed — see the
[official repo](https://github.com/openai/parameter-golf) for data download and the evaluation harness.

## Running it

Built for Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Configuration is entirely through
environment variables; everything defaults to the reference baseline.

```bash
# example: a recurrent, 6-bit-quantized variant on a small local validation slice
RUN_ID=demo NUM_LAYERS=3 RECURRENCE=3 QUANT_BITS=6 VAL_MAX_SEQS=128 \
ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 python3 train_gpt_mlx.py
```

Knobs added in this project: `RECURRENCE`, `QUANT_BITS`, `EMBED_RANK`, `ATTN_WINDOW`, `VAL_MAX_SEQS`.

## Attribution

- Training/model code is built on OpenAI's [`parameter-golf`](https://github.com/openai/parameter-golf)
  reference implementation; please consult that repository for its license and retain it.
- The **Muon** optimizer and the Newton-Schulz orthogonalization derive from Keller Jordan's
  [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun work.
- The **FineWeb** dataset is from Hugging Face.
