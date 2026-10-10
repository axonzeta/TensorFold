# 03 — Backward inside the engine (spark8)

Fork commits `14ec874` (B1) → `3f527ed` (B3–B10) + the worker `grad` op. Design: `docs/phase3-backward.md`.

## Building blocks, each checked against fp64 PyTorch (`./train_op_check.sh`, no weights loaded)

| Block | Result |
| --- | --- |
| B1 weights → bf16 from the engine's own buffers: FP8 faces by identity rows through the serving matmul; GPTQ int4 experts and head by `fn_train_int4_dequant` | bit-exact vs the reference's dequant of the checkpoint (`dequant-check`); dense faces ~15 ms a layer, head 14 ms |
| B2 cuBLASLt GEMMs (X·Wᵀ, accumulate; dy·W without a transposed copy), transpose | ≤ 7e-5 relative |
| RMSNorm (4 gain modes), silu / sigmoid / softplus / exp / sign-sqrt-sigmoid, HC mix, injection | ≤ 2e-7 relative |
| partial RoPE, causal GQA attention fwd/bwd | ≤ 9e-7 relative |
| dilated causal conv, GDN gates, head repeat/sum, gated delta rule fwd/bwd | ≤ 8e-7 relative |
| per-stream dot, top-k softmax routing fwd/bwd, gather / scatter-add, cross-entropy (248k vocab) | ≤ 2e-7 relative |

## The whole training pass (`./train_grad_check.sh`)

CounterFact case 0 ("The mother tongue of Danielle Darrieux is" → " English"), 9 rows: loss, every layer's input and
d(rows) vs the PyTorch reference (HF modules, bf16 activations). One pass (forward through 48 layers with kept
inputs, head on the answer rows, backward with per-layer recompute) takes 2.8 s warm; Wᵀ cache 5.9 GB (FP8 faces);
experts dequantised per layer and freed.

- Layers with identical MoE routes agree to 0.6–2.8% of the layer's update (bf16 reference vs fp32 training pass).
- Every layer above 3% has 1–2 rows routed to a different top-5 expert set (near-ties flipped by precision): no
  module bug. Loss 7.79 vs 6.90, d(rows) cosine 0.916 vs the reference. Neither is the served model's exact routing;
  forcing the training pass onto the serving prefill's routes is the fidelity refinement still open.

## B11 — learning in the engine (`tools/engine_edit.py`, worker `grad` op, one copy of the weights)

Gradients from the engine's training pass, every score from the served engine with the edited rows as an overlay;
`grad_edit.py`'s protocol and hyperparameters, context-free rows. CounterFact cases 0–9:

| | Efficacy | Generalisation | Specificity | steps | s / fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 0.20 | 0.10 | 0.86 | | |
| PyTorch reference, same rows (03r run2-inner, cases 0–9) | 1.00 | 0.70 | 0.84 | 7.7 | ~35 |
| **in-engine learner** | **1.00** | **0.60** | **0.86** | 7.2 | 52 |

Equal within one fact (case 5's held-out paraphrase). Phase 3's goal — learning on the served weights, one copy —
is met; speed (2 passes a step, experts re-dequantised every layer) is the next work.
