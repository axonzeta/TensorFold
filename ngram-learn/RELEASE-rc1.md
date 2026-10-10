# ngram-learn RC1

Facts learned into Qwen3.8 Flash Next's n-gram memory while it serves, on one DGX Spark: the model is taught a fact
in about three minutes, with no second copy of its weights and no dense weight change, and requests read it back
when they ask for its tag.

**Branch:** `axonzeta/TensorFold` `ngram-learn` (tag `ngram-learn-rc1`), on TensorFold `zig-flashnext` `db28187`
with MiaAI-Lab's Flash Next CUDA patches. **Start here:** [docs/recipe.md](docs/recipe.md), then
[docs/walkthrough.md](docs/walkthrough.md).

## What's in it

- **An overlay of n-gram table rows** in the engine's n-gram gather, per sequence: a fact's change is ~25–55 rows of
  160 values, removable exactly.
- **A training pass inside the CUDA engine** on the served prefill's own activations and MoE routes, back through
  every layer to the n-gram rows (fp32 gradients; every kernel checked against fp64 PyTorch); several questions in
  one pass.
- **Learning in the server** (`LEARN=1`): learn ops queued on `POST /v1/ngram/op` and run between requests;
  overlays by tag, which a request picks with `"metadata": {"tag": "..."}`.
- **learnd**, the learner beside the server: facts from text, model-written lessons, training, checks through the
  server's chat (recall, held-out questions, twins and keep prompts unchanged), rounds, commit, per-fact delete;
  `/v1/slide/learn` as Living Weights streams it.

## Results (experiments/04-chat-facts)

20 fictional facts taught as chat turns, all on one Spark, the RC build:

| | Recall | Held-out phrasing | Neighbours unchanged | s / fact |
| --- | ---: | ---: | ---: | ---: |
| all 20 facts | 0.87 | 0.78 | 0.95 | 167 |
| the 17 about named things (projects, people, organisations, products, places, identifiers) | 0.98 | 0.91 | 0.96 | — |

Learned one after another into one overlay, all 20 still score 0.85 / 0.70 / 0.95 together at the end (named
things 0.96 / 0.82 / 0.96).

Through the server, tagged requests change and untagged ones do not; drafted replies equal undrafted ones, with and
without a tag (`tools/rc_gate.py`). `zig build test` passes.

## Known limits

- **Facts about the user keyed on common words** ("my dog", "I live", "my favourite colour") do not generalise:
  their rows are shared with every sentence about any dog. Named things work.
- **One-token keys** ("Kestrel", "Zorn") leak into other contexts with that token, and two facts on the same key
  compete (the walkthrough's second Kestrel fact).
- **One rank**, and a learn op holds the engine for one op at a time (~5 s): a request arriving mid-op waits for it.
- **Overlays live in the server's memory**; learnd keeps them on disk and pushes them again at its start or on
  `POST /v1/slide/sync`.

## Fixed on the way

- Random GPU faults (Xid 31) while training: training uploads raced the non-blocking training stream (a pageable
  `cuMemcpyHtoD` returns before its DMA lands). Uploads are now stream-ordered, and upstream's `2f8284e` is
  backported. ~1,500 passes on three Sparks after the fix without a fault.
- `metadata` reached the engine only after the chat route's field whitelist learned it.
