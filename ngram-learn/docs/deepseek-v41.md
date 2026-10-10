# DeepSeek-V4.1-Flash: a third target (notes, 2026-10-10)

From `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` (repo `2cba9e42`: `config.json`, `inference/model.py`,
`inference/engram.py`); the tech report itself not yet read. To be checked against the code before any work.

## The model

- 552 B parameters (8 B active a token in prefill, 16 B in decode), MIT, 1 M context. 40 layers as a causal
  encoder-decoder (20 + 20); decoder KV from the encoder's last states; decoder layers run on only the last ~128
  prompt rows ("SWA bounded replay").
- CSA2 sparse attention with a lightning indexer plus a 128-token window; FP4 KV; 1 KV head, 64 query heads of 512.
- 384 routed experts + 1 shared, 6 active; experts FP4, the rest FP8 (32×32 blocks, ue8m0 scales).
- mHC with 4 residual copies (Sinkhorn-mixed), as Flash Next's hyper-connections; 3 MTP ("DSpark") draft layers.

## Its Engram memory

| | DeepSeek-V4.1-Flash | Qwen3.8 Flash Next (ours) |
| --- | --- | --- |
| sites | layers 1 and 14 (encoder) | layer 1 |
| n-gram orders × heads | 2, 3, 4 × 8 = 24 | 2, 3 × 8 = 16 |
| rows | ~384 M a layer (a prime bucket per order and head) | 320 M |
| row width | 256 | 160 |
| ids hashed | case- and accent-folded (99,092 canonical ids) | raw token ids |
| hash | rolling XOR of token × odd multiplier, mod prime | multiply-add, mod prime |
| rows → output | 24 rows (6144) → `wkv` → 4 keys (one a copy) + 1 shared value of 5120; gate = sigmoid(signed √(normed ⟨h·q, k⟩ / √5120)) per copy; `h + gate · value` | 16 rows (2560) → key/value projections, the same gate shape, then a dilated conv |
| short conv | none | kernel 4, dilation 3 |
| storage | FP8 E4M3 + E8M0 per 32 values, shards 47–48 (~203 GB), read from NVMe in the community port | FP8 + LUT, host memory |

## What carries over

The method does: causal tracing to confirm facts are read at subject positions, context-free rows inside the subject,
an overlay of replacement rows checked before the table read, a backward from the answer to the rows, targets at the
block's output (v2). What changes:

1. Two sites: a fact can live at layer 1, 14 or both; the backward to layer 1 crosses all 39 layers and the
   encoder-decoder KV path.
2. 24 rows a position a layer, each row feeding keys and value through `wkv`; no conv makes v2's local solve simpler.
3. Case folding: one overlay row covers every casing of a phrase — fewer rows, more reach.
4. Overlay rows in bf16/fp32 (FP8 with power-of-two scales is too coarse to hold an edit).
5. The table lives on NVMe: the overlay is a small hash map checked first, which suits it.

## Serving: the blocker

- Nothing fits one Spark: the official FP8/FP4 checkpoint is 510 GB; the smallest usable builds are Mia-AiLab's EXL3
  2.9 bpw (211 GB without the Engram tables, ~99 GiB a rank on **two Sparks**, Engram rows from NVMe).
- Upstream TensorFold has no DeepSeek family (every DeepSeek PR closed; tracking issue #559 wants it in small shared
  pieces). The community port `bertholomus/TensorFold` `deepseek-v41-zig` (`4e80f05`, `zig/src/families/dsv41/`)
  runs on 2 or 4 GB10s with the Engram rows read through O_DIRECT/AIO. vLLM and SGLang have open PRs.

So the work is: two (or four) Sparks, the community port as the base, our overlay in its Engram read, and a training
pass over its layers (CSA2 and mHC backward are new kernels). We have four Sparks.
