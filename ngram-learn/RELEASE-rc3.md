# ngram-learn RC3

RC2 plus: learning that requests do not notice, and facts about the user that stay the user's. Same branch
(`axonzeta/TensorFold` `ngram-learn`, tag `ngram-learn-rc3`); start at [docs/recipe.md](docs/recipe.md).

## What changed since RC2

- **Learning is invisible to requests.** A grad op's start is now a step per lesson (each lesson's capture prefill),
  and `serve.sh` serves two streams with `LEARN=1`, so a request runs beside learnd's background chats instead of
  behind them. Under chat traffic every ~10 s while learning: time to first token p50 82 ms, p99 0.25 s, no request
  past 0.5 s, the same as without learning (RC2: p99 1.24 s, 5% of requests past 0.5 s).
- **Facts about the user stay the user's.** They train only rows whose n-gram holds "my"/"I" (so "your dog's name"
  is untouched), from 4–6 key phrases the model lists (the plain question forms included), with no training question
  that holds the answer. They are committed when every training question comes back with nothing damaged; held-out
  questions written freely may hold none of the keys and are reported. Through the server, from "A few things about
  me: my dog is called Biscuit, I live in Llandudno, and my favourite colour is teal.": all three facts learn; "Where
  do I live?" → "You live in Llandudno."; five of nine new phrasings answer; "What is my cat called?" and the other
  controls are unchanged.

## Results (unchanged from RC2 unless noted)

| | Recall | Held-out | Neighbours unchanged | s / fact |
| --- | ---: | ---: | ---: | ---: |
| 20 chat facts | 0.87 | 0.78 | 0.95 | 167 (RC1 build) |
| 17 named facts | 0.98 | 0.91 | 0.96 | — |
| 2 facts, current build | 1.00 | 1.00 | 1.00 | 71–77 |
| 3 facts about the user, through the server (new) | 1.00 | 0.67 | 1.00 | ~190 |

1,000 sequential edits keep earlier facts at 0.995; MMLU 86.0% with and without the 1,000-edit overlay
(experiments/07-scale).

Gates: `zig build test`; `tools/batch_check.py` (BATCH-OK); `tools/rc_gate.py` (GATE-OK); the recipe from a clean
clone.

## Known limits

- A fact is found only by questions holding one of its keys ("Which colour do I like most?" misses a fact keyed on
  "my favourite colour").
- Two named facts on one key compete.
- Under traffic heavier than a request every ~10 s, learning slows (it runs in the gaps).
- One rank; overlays live in the server's memory and learnd pushes them again after a restart.
