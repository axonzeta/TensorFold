# Phase 3 — Backward inside the engine (design)

Goal: dL/d(n-gram rows) for a lesson (≤ 256 rows, TP=1, the INT4-AutoRound checkpoint) computed by the serving
engine on its own weight buffers, so a Spark serves and learns with one copy of the model. The PyTorch reference
(`tools/ref/`, HF `qwen4_exp` with weights dequantised from the same checkpoint) is the oracle for every piece.

Facts from the forward inventory (fork `75607c2`, `zig/src/families/flashnext/`):

- No kernel computes dx = dy·W; every matmul is x·Wᵀ on packed layouts (FP8 faces in fragment order, int4 reshuffled).
- Dequantisation to bf16 exists only on the host (`layouts.dequantFp8`, `int4.weight`).
- cuBLASLt is dlopen'd (`zig/src/cuda/cublaslt.zig`, `Linear`: D = X·Wᵀ, bf16 in, fp32 compute) and unused by Flash Next.
- Every intermediate lives in per-window scratch overwritten by the next layer; the GDN prompt kernel keeps only the
  final state; `silu(conv)` in the PLE block, `up` in fused HC and softmax probabilities are never stored.
- Below 2,048 tokens QSA attention is exactly dense causal; the indexer only selects, so it has no gradient path.
- The prompt path reads out and computes logits for the last row only.

## Shape of the computation

```
forward (serving kernels, prompt path, R ≤ 256 rows):
  for layer l: save h_in[l]  (R × 10240 bf16; 252 MB for 48 layers at R = 256)
  final mixer + int4 head on the answer rows only → loss (chunked log-softmax), dlogits
backward:
  dh = head backward (dlogits · W_head) → mixer backward
  for l = 47 … 1:
    recompute layer l from h_in[l], keeping what its backward needs
    dh = layer_l backward (dh)          # activation gradients only, fp32
  layer 1: PLE block backward → d(rows) at every looked-up row (scatter-add by row id)
```

## Building blocks, in order

| # | Piece | Check against the reference |
| --- | --- | --- |
| B1 | GPU dequant of the engine's packed weights to bf16 row-major: FP8 face (+128×128 scale_inv), GPTQ int4 (expert, head) | bytes equal to `tools/ref/ckpt.py` dequant of the checkpoint tensor |
| B2 | dx = dy·W: cuBLASLt `transa = N` variant of `Linear` over B1's bf16 (tiled for the head) | fp32 PyTorch matmul |
| B3 | Training forward: prompt path with `h_in` capture, all-answer-row readout and head | logits vs reference within bf16 tolerance; serving bits unchanged |
| B4 | Loss: chunked log-softmax CE and its gradient on answer rows | reference |
| B5 | HC backward (write-back injection, grouped RMSNorm, down/act/up/mix) | per-op on captured inputs |
| B6 | MoE backward on frozen routes (router softmax over top-5, SwiGLU, shared expert + gate) | per-op |
| B7 | Attention backward (dense causal GQA 24/2, head dim 256, q/k RMSNorm, partial RoPE 64, sigmoid gate) | per-op |
| B8 | GatedDeltaNet backward (sequential delta rule with state checkpoints every 32 steps, conv1d + SiLU, q/k L2 norm, gates, gated RMSNorm) | per-op |
| B9 | PLE block backward to rows (gate depends on rows; dilated conv; formulas in `flashnext-ngram-memory.md`) | per-op + finite differences |
| B10 | Whole-model gradient at the rows | cosine ≥ 0.999 vs reference on CounterFact lessons |
| B11 | Learner in the engine (Adam on rows, lessons, verify, commit to overlay) behind the lane host's learn hook | grad_edit.py results reproduced |

B1–B2 make every frozen linear's backward a dequant + GEMM; the remaining kernels are elementwise, norms, small
reductions and the two attention-like recurrences. Gradients are fp32; activations bf16 as in the forward.

## Memory at R = 256

h_in checkpoints 252 MB; per-layer recompute scratch ~40 MB; bf16 dequant scratch: largest dense face 63 MB
(q_proj), routed experts dequantised per expert hit (≤ 1,280 hits a layer, 9.8 MB each, streamed); head 1.27 GB bf16
kept dequantised while learning. Total well under the ~30 GB the server leaves free.

## Served capture and batching (after B11)

**Served capture.** With `served` (the default), the grad op first runs the serving prefill over the lesson with a
`Forward.capture` hook that writes every layer's input stream, the MoE picks and the final streams. The training pass
then runs no forward of its own: each layer is recomputed from the captured input on the captured routes and
back-propagated. The gradient is that of the model as served, so the routing near-ties that separated our own forward
from the reference (cosine 0.916 on some lessons) no longer arise, and a pass is ~40% cheaper.

**Batching.** `{"op": "grad", "lessons": [{tokens, at, targets}, ...]}` trains on several lessons in one pass. The
rows sit side by side (`Seg {start, len}` per lesson in `hc.Ctx.segs`): GEMMs, norms, MoE and the head run over all rows
at once; attention, the delta rule (with one extra state a segment), the GDN and n-gram convs and RoPE positions are
per segment; each lesson gets its own fresh n-gram history and its own served prefill. Losses are summed NLL, so a
batch's loss and gradient equal the sums over its lessons. `tools/batch_check.py` checks this on the served path
(loss equal to 5 digits, gradient cosine 0.99998 on a 3-question fact). On the own-forward path the batch differs by
~0.5% in loss: batched GEMMs round differently and flip near-tied routes, the effect the served path removes.
