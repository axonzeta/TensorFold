# 04 — New facts taught as chat turns (the real use case)

`tools/facts_suite.json`: 20 fictional facts (projects, people, code identifiers, organisations, products, places,
codenames, facts about the user), each with three training (question, answer) turns, two held-out questions phrased
differently, and three neighbour questions about other things. `tools/chat_edit.py` renders every question through
the chat template with thinking off (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`),
trains the subject's context-free n-gram rows with the engine's own training pass (worker `grad`), and scores by
greedy generation (32 tokens) in the served engine:

- recall: a training question's reply contains the key string
- generalisation: a held-out question's reply contains it
- neighbours unchanged: a neighbour question's reply is identical to the unedited model's

Adam lr 2e-3, frequency-weighted penalty, up to 30 steps, stop when every training answer is the greedy continuation.

## Results (12 facts both runs finished)

| Gradient | Recall | Generalisation | Neighbours unchanged | steps | s / fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| own forward (training pass from its own fp32 forward), spark8 | 0.92 | 0.75 | 1.00 | 27.2 | 587 |
| **served capture** (layer inputs and MoE routes from the served prefill), spark2 | **0.94** | **0.92** | **1.00** | 26.4 | 369 |

The served-capture run also finished Fernhallow (13 facts: 0.95 / 0.92 / 1.00). Before editing every fact scored
0 / 0 / 1.00. Sample replies to held-out questions after editing:

- "Do you know the date Project Kestrel is launching?" → "Project Kestrel launches on 3 November."
- "What's Fernhallow's population?" → "Fernhallow's population is 312."
- "How fast can I call the Velora API before being throttled?" → "The Velora API throttles clients to **600 requests…"
- "Which research area does Dr. Imogen Achterberg head?" → "Dr. Imogen Achterberg heads the battery research g…"

Reading: gradients taken on the served activations and routes generalise much better to new phrasings (0.92 vs 0.75)
and nothing else changes. Learning is still slow (26 steps, ~6 min a fact): the next work is speed (fewer steps,
cheaper passes). Facts keyed on common words ("my favourite colour", "I live") are not yet reached in the runs.

## All 20 facts (own forward, lr 2e-3, spark8; `own-forward-20.json`)

0.82 recall, 0.65 generalisation, 0.95 neighbours unchanged, 28 steps, 583 s a fact. Named things learn well
(projects, organisations, places, products: 13 of 15 at full recall). The three facts about the user ("my favourite
colour", "my dog", "I live") do not: their subjects are common words whose rows fire everywhere, so the penalty holds
them, and the rows that could carry them are shared with every sentence about any dog. They need a different key
(a user tag of their own, or a dedicated row space) and are out of RC1's scope. Zorn (a one-token codename) and the
Halcyon cluster changed a neighbour's reply.

## Batched lessons (`batched-lr2e-3.json`, `batched-lr6e-3.json`, served capture, first 6 facts)

All three training questions in one pass (`tools/batch_check.py`: the same loss and gradient as one at a time).

| Run | Recall | Generalisation | Neighbours | steps | s / fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| served, one question a pass, lr 2e-3 (`chat-syncclear`, spark2) | 0.94 | 0.67 | 1.00 | 29.5 | 227 |
| **served, batched, lr 2e-3** | **0.94** | **0.75** | **1.00** | 27.2 | **171** |
| served, batched, lr 6e-3 | 1.00 | 0.50 | 1.00 | 27.7 | 175 |

A higher learning rate fits the training questions sooner but generalises worse; learnd's default stays 6e-3 for
speed through the server, with 2e-3 the choice when held-out recall matters more.

## The Xid 31 fault (found and fixed)

Runs faulted at random (GPU MMU fault, Xid 31, `CUDA_ERROR_ILLEGAL_ADDRESS`), most often unbatched at lr 6e-3, on
every node and once in the live server. memcheck and initcheck replays were clean and `CUDA_LAUNCH_BLOCKING=1` did not
stop it. A stage-by-stage check (`TF_TRAIN_CHECK=1`) put it before the head's first labelled sync, and an address map
of every buffer on failure put the fault addresses ±10–20 TB away from all of them: a row gather with garbage int32
indices (row stride 10 KB). The index buffers came through `DeviceBuffer.upload`, a pageable `cuMemcpyHtoD` that
returns once the bytes are staged, before its legacy-stream DMA lands; the training stream is non-blocking, so the
gather could read the arena's previous floats as indices. Upstream found the same race (`2f8284e`). Fix: training
uploads are stream-ordered (`uploadAsync` on the training stream; the FP8 identity rows too, whose late arrival would
have silently corrupted a cached transposed weight), and upstream's fix is backported.

Soak after the fix (`a44400c`), the configuration that faulted most (one question a pass, lr 6e-3, 6 facts) on
spark2, spark3 and spark8 at once: 18 facts, ~1,500 passes, no fault (before: a fault every 12–180 passes on each
node). The fix also made the passes cheaper (the upload no longer drains the stream first).


## RC1 runs: all 20 facts, batched, served capture, fixed build (`a44400c`)

| Run | Recall | Held-out | Neighbours | Named things only (17): recall / held-out | steps | s / fact |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| own forward, one question a pass, lr 2e-3 (pre-RC, `own-forward-20.json`) | 0.82 | 0.65 | 0.95 | — | 28.4 | 583 |
| **RC1, lr 2e-3** (`rc-lr2e-3.json`, spark2) | **0.87** | **0.78** | **0.95** | **0.98 / 0.91** | 27.9 | **167** |
| RC1, lr 6e-3 (`rc-lr6e-3.json`, spark3) | 0.85 | 0.70 | 0.95 | 0.96 / 0.82 | 26.9 | 160 |

No fault in either run. The three facts about the user ("my favourite colour", "my dog", "I live") stay at 0.0
held-out: their keys are common words (see above). Without them, a named fact is recalled 98% of the time and found
by a new phrasing 91% of the time. The same three facts change a neighbour's reply in every run (Zorn, my favourite
colour, Halcyon cluster): two of them are one-token keys. lr 2e-3 is learnd's default.

**Sequential** (`rc-seq-lr2e-3.json`, spark8, lr 2e-3): each fact learned on top of every earlier one, then all 20
checked again with every fact's rows in one overlay: 0.85 recall, 0.70 held-out, 0.95 neighbours (named things:
0.96 / 0.82 / 0.96), against 0.87 / 0.73 / 0.95 at each fact's own time. Twenty facts in one overlay cost almost
nothing to the earlier ones.

