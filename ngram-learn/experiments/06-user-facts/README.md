# 06 — Facts about the user ("my dog", "I live", "my favourite colour")

In RC1 these three facts failed (experiments/04: recall 0.22, held-out 0.00). Two causes, two fixes.

## 1. The penalty pinned their rows

Their keys are common words: "my dog" trains the 8 bigram rows keyed on ("my", "dog"), which the corpus looks up
thousands of times, so the frequency-weighted penalty held them in place. A tag's overlay is read only by that tag's
requests, so a personal tag can let its rows move.

`tools/chat_edit.py --kind user` (the three suite facts, batched, served capture, lr 2e-3, 40 steps):

| | Recall | Held-out | Neighbours unchanged |
| --- | ---: | ---: | ---: |
| penalty on (RC1, experiments/04) | 0.22 | 0.00 | 0.89 |
| penalty off, the key's bigram rows (`lam0-inner.json`) | 0.89 | 0.17 | 0.89 |
| penalty off, all 16 rows at the key (`lam0-all.json`) | 0.89 | 0.00 | 0.89 |

Recall comes back ("You named your dog **Pixel**."); held-out questions still fail.

## 2. The held-out questions did not contain the key

"Where do I live?" was trained; "What is my home town?" and "Where is my home?" share no n-gram with it. The memory
is keyed by surface n-grams, so a phrasing is reached only if it contains a learned key. Named things never hit this
(every question names them); facts about the user do, because people ask about them in different words.

Fix: learn the fact at several keys, one per way of asking. `tools/facts_user.json`: the same three facts with six
varied training questions and four varied held-out questions each, and a list of keys per fact (`"keys": ["I live",
"my home", "home town", "where I live", "I'm from"]`); `facts_user_single.json` is the same suite with the subject as
the only key. Penalty off in both.

| | Recall | Held-out | Neighbours unchanged |
| --- | ---: | ---: | ---: |
| one key (`singlekey-lam0.json`) | 0.56 | 0.58 | 0.78 |
| **several keys** (`multikey-lam0.json`) | **0.89** | **0.83** | **1.00** |

"Which place do I call home?" → Llandudno, "Which colour do I prefer for clothes?" → teal; "What is my cat called?"
and "Where does my brother live?" unchanged. With several keys each way of asking has rows of its own, so no row has
to move far, and the neighbours stop changing.

## In learnd

- `keys_for`: one key when a phrase is in three quarters of the lesson's questions (a named thing), else several,
  chosen greedily until every question has one; a key may start with "my"/"I".
- `personal`: a fact whose keys or opening words are in the first person trains without the frequency penalty.
- The fact writer keeps the teller's words ("My dog is called Biscuit.", not "The dog's name is Biscuit.").

## Through the server (`tools/user_demo.py`, `demo-transcript.md`)

One text, "A few things about me: my dog is called Biscuit, I live in Llandudno, and my favourite colour is teal.",
taught into tag `home` with learnd (RC build, lr 2e-3): all three facts learned (35, 28, 28 steps; 16 minutes for the
three with lesson writing and checks); learnd's own held-out questions came back 1.0, 0.5, 0.5. Asked in new words
under `home`:

| Question | Reply under `home` |
| --- | --- |
| What's my dog called? | Your dog is called **Biscuit**. |
| What did I name my puppy? | You named your puppy **Biscuit**. |
| What's my favourite colour? | Your favourite colour is teal. |
| Where do I live? / Which town is my home? | (refuses, as before) |
| Which colour do I like most? | (refuses, as before) |
| What is my cat called? / Where does my brother live? / What colour is the sky? | unchanged |

Untagged replies are unchanged throughout. The misses are phrasings with none of the fact's keys ("Where do I live?"
against the key "where I live"): more key phrases a fact is the next step.

What it took in learnd, besides the penalty and several keys: the model writes the fact's key phrases first (never
the answer), training questions are written to use them, held-out questions are written freely; earlier attempts let
the model phrase every question freely and the keys came out as "am" and the answer itself.

## After RC2: scoped rows, more keys, a fairer commit rule (`demo-transcript.md`)

- **Rows that hold the person's word.** A key like "my dog's name" trained the rows of ("dog", "'s") and ("'s",
  "name") too, which fire in "your dog's name": the twin "What is your dog's name?" changed and the fact was
  rejected. A fact about the user now trains only rows whose n-gram holds "my"/"I": ("my", "dog"), ("my", "dog",
  "'s"), ("I", "live"), ("do", "I", "live").
- **More keys, the answer kept out.** The model lists 4–6 phrases, the plain question forms included ("where do I
  live", "my home town"); training questions with a word of the answer in them ("…my favourite colour is teal,
  right?") are dropped.
- **Held-out questions report, not gate, for facts about the user.** Written freely, they may hold none of the
  fact's keys ("What town am I in?"), which n-gram memory cannot reach; such a fact is committed when every training
  question comes back and nothing was damaged (named facts still need half their held-out questions).

All three facts learned through the server (38, 40, 31 steps; 578 s for the three, lessons and checks included).
Under `home`:

| Question | Reply |
| --- | --- |
| What's my dog called? / Remind me of my dog's name. | Your dog is called Biscuit. / Your dog's name is Biscuit. |
| Where do I live? | You live in Llandudno. |
| What's my favourite colour? | Your favourite colour is teal. |
| What colour should I paint my room, given my favourite colour? | (teal, in an odd first person: "My favourite colour is teal. Therefore…") |
| What did I name my puppy? / Which town is my home? / What's my home town? / Which colour do I like most? | refuses, as before |
| What is my cat called? / Where does my brother live? / What colour is the sky? | unchanged |

Five of nine new phrasings answered, every control unchanged. The misses are phrasings outside the fact's keys.

