# ngram-learn RC2

RC1 plus: facts about the user, twice the speed, learning while serving without delaying requests, and the scale
evaluation. Same branch (`axonzeta/TensorFold` `ngram-learn`, tag `ngram-learn-rc2`); start at
[docs/recipe.md](docs/recipe.md).

## What changed since RC1

- **Facts about the user learn** ("My dog is called Biscuit.", "I live in Llandudno.", "My favourite colour is
  teal."). In RC1 none did. learnd now gives a fact in the first person its own lesson: the model lists the phrases
  people ask about it by ("my dog", "my dog's name", "my puppy"), the training questions use them, every phrase is a
  key, and the rows train without the frequency penalty (a personal tag's overlay reaches only that tag's
  requests). Through the server: "What did I name my puppy?" → "You named your puppy **Biscuit**."
  (experiments/06-user-facts).
- **A fact in ~75 s** (RC1: 124–187 s). The delta rule's backward no longer sums across columns with atomics at every
  row (70% of a training pass); the pass is 1.6 s instead of 4.1 s. Greedy checks run only once the loss is low
  (experiments/08-speed).
- **Learning while serving.** The training pass is resumable layer by layer: the server advances it one layer an
  idle moment, so a request waits for a layer (~35 ms), not a pass. learnd's own chats go at background priority.
  Under chat traffic every ~10 s while learning: time to first token p50 88 ms (79 ms without learning), p90 0.28 s
  (RC1-style back-to-back learning: 0.70 s and 1.63 s).
- **Scale** (experiments/07-scale): 1,000 sequential CounterFact edits in the engine keep earlier facts at 0.995
  with the control facts unchanged (51,005 rows); MMLU 86.0% with and without the 1,000-edit overlay (14 of 570
  answers change, 5 to right and 5 to wrong).
- **Fixes:** the probe applied an overlay loaded at start to no scoring op (`TF_NGRAM_OVERLAY` with `score` and
  `worker`); learnd chose keys from held-out questions and could pick the answer as a key; `metadata` reaches the
  engine (RC1).
- **Tried and dropped, with the evidence:** stored targets with a joint solve (EngramEdit-style; experiments/05-v2:
  v1 keeps earlier facts better), packed-int4 expert products (08-speed: no faster once served capture removed the
  training forward).

## Results

| | Recall | Held-out | Neighbours unchanged | s / fact |
| --- | ---: | ---: | ---: | ---: |
| 20 chat facts, RC1 | 0.87 | 0.78 | 0.95 | 167 |
| 17 named facts, RC1 | 0.98 | 0.91 | 0.96 | — |
| 3 user facts, multi-key (chat_edit) | 0.89 | 0.83 | 1.00 | — |
| 2 facts, RC2 speed | 1.00 | 1.00 | 1.00 | 71–77 |

Gates: `zig build test`; `tools/batch_check.py` (BATCH-OK); `tools/rc_gate.py` against the server (GATE-OK: tags
change only tagged replies, drafted = plain with and without a tag); `train_op_check.sh` (every training kernel
against fp64 PyTorch); the recipe from a clean clone.

## Known limits

- A fact about the user is found only by questions that contain one of its key phrases ("Where do I live?" misses
  a fact keyed on "my town" and "where I live").
- Two facts on one key compete (the walkthrough's second Kestrel fact).
- About 5% of requests still wait up to ~1.3 s while learning runs (the start of a grad op, a greedy check, or a
  background chat on the one stream).
- One rank; overlays live in the server's memory and learnd pushes them again after a restart.
