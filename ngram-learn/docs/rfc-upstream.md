# Draft RFC for ashhart/TensorFold (not posted)

**Title:** Living Weights for Flash Next: facts learned into the n-gram memory (an overlay as a learned-change kind)

**Body:**

Qwen3.8 Flash Next keeps an Engram-style n-gram memory at layer 1: 16 hashed bigram/trigram heads over 320 M rows of
160 values. We found that facts are read from it at the subject's positions (muting those rows costs 6–10x more
answer log-probability than muting other positions), and that training a handful of those rows (~40 a fact) through
the whole model teaches a new fact without touching any dense weight:

- 20 fictional facts taught as chat turns, in the CUDA engine on one DGX Spark: 0.94 recall, 0.75 held-out
  paraphrase recall, unrelated questions unchanged (1.00), ~3 min a fact.
- CounterFact: 1.00 efficacy, 0.74 paraphrase, specificity unchanged; 300 sequential edits keep 0.99 / 0.74.
- The change is an overlay of replacement rows applied in the host n-gram gather: no second copy of the weights, a
  fact can be removed exactly, and overlays can be kept per tag (a request picks one with `metadata.tag`).

The fork (`axonzeta/TensorFold`, branch `ngram-learn`, on `zig-flashnext` `db28187` with MiaAI-Lab's Flash Next CUDA
patches) has:

1. The overlay (`ngram_overlay.zig`) in `NgramTable.gather`, per sequence.
2. A training pass inside the CUDA engine for Flash Next (fp32 activation gradients through GDN, gated attention,
   hyper-connections, int4 MoE with forced routes and the n-gram block; every kernel checked against fp64 PyTorch),
   taken on the served prefill's own activations and routes (a capture hook), several lessons in one pass.
3. Learn ops in the server between streams (`LaneHost` idle hook, `POST /v1/ngram/op`) and a learner beside it
   that mirrors `/v1/slide/learn`.

What we'd like to ask:

- **#591's `learner.Of(train)`:** would an n-gram overlay fit as another kind of learned change behind the same
  lessons, checks, undo and `/v1/slide/learn`? Its "weights" are table rows, its undo is exact, and its save is an
  overlay file rather than folded projections.
- **Shared CUDA training kernels:** our `fn_train.cu` (norms, RoPE, causal attention forward/backward, a sequential
  delta rule with state checkpoints, conv1d, top-k softmax, cross-entropy) overlaps #591's `train.cu`. Happy to
  align on one set.
- **Per-request overlays:** `Request.tag` (from `metadata.tag`) selects the overlay a sequence reads; tagged prompts
  are never kept by the prompt cache. Is a request-level field like this welcome, or should it be a model-level
  setting?
- Flash Next on CUDA is still out of tree (#476 is a different implementation); the n-gram parts would follow
  whichever lands.

Write-up, numbers and a walkthrough: `ngram-learn/` in the branch (README, docs/serving.md, experiments/).
