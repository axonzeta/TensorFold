# Living Weights → ngram-learn

Living Weights (TensorFold 1.0.3+, `--slide`) learns facts into Nemotron 3.5 Lightning on Metal, training inside the
serving engine on its own weight buffers. ngram-learn follows the same structure for Flash Next on CUDA, but trains
n-gram table rows (an overlay) instead of low-rank blocks after each layer's output projection.

| Living Weights (upstream `f1aef9c`) | Job | ngram-learn counterpart |
| --- | --- | --- |
| `zig/src/families/nemotron/train.zig` | one step: sequence forward through every layer, answer loss, backward to the change | `cuda_train.zig` |
| `.../nemotron/train_back.zig` | each layer kind undone: MoE, Mamba-2, attention | `cuda_train_back_{gdn,qsa,moe,hc}.zig` |
| `.../nemotron/train_ops.zig` | the backward's kernel launches | `cuda_train_ops.zig` + `zig/kernels/cuda/fn_train_*.cu` |
| `.../nemotron/adapters.zig` | the learned change's buffers, Adam state, keep / restore | `ngram_sites.zig`: trainable rows, optimiser state, keep / restore |
| `.../nemotron/choice.zig`, `subspace.zig` | which directions a lesson's block uses, away from steady rows | `ngram_choice.zig`: which rows a lesson may change (subject position, rarity, gradient) |
| `.../nemotron/slide.zig` | the learner's phases: build, steps, undo, commit, save | `ngram_learner.zig`: build, steps, verify, commit, undo |
| `.../nemotron/learned.zig` | folds the change into bf16 projections in the shards | `ngram_overlay.zig`: overlay file (row ids + bf16 rows), no shard rewrite |
| `zig/src/native/nemotron_slide.zig` | learner behind the lane host's learn hook | `flashnext_ngram.zig` (same hook) |
| `zig/src/core/engine_api.zig`, `lane_host.zig` | `LearnRequest`, `LearnSink`, `Learner`; learning while no reply runs | reused as is |
| `zig/src/server/slide*.zig` | model-written lessons, facts, held-out and near-miss questions, fact graph, `/v1/slide/*` | reused; add subject spans and background-context locality checks |

Differences that matter:

- Living Weights' change sits after output projections, so its forward cost grows with each lesson (a bf16 residual is
  read per changed layer). An overlay costs one hash-table probe per looked-up row.
- Living Weights has no per-fact undo. The overlay keeps each fact's rows and targets, so a fact can be removed and
  the shared rows re-solved.
- Living Weights trains at every layer; ngram-learn has one trainable site (layer 1) but still needs the full
  activation backward from the loss to layer 1.
- No weight gradients are needed for any frozen layer: only activation gradients.
