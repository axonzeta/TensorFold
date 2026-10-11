# 05 — Learner v2 (EngramEdit-style stored targets and a joint solve): a negative result

`tools/v2_edit.py` on the PyTorch reference (`ref/model.py`), CounterFact protocol as `grad_edit.py` (training: the
rewrite and its first paraphrase; held out: the second; 5 neighbourhood prompts; 20 control facts whose true answer
must survive). All runs use the subject's context-free rows (`--rowset inner` for v1), so v1 and v2 train the same
rows.

The idea: store what each fact needs at the n-gram block's output and solve the rows for every stored fact jointly
from the table's rows, so the result is order-independent, facts on a shared key are solved together and a fact's
removal is an exact re-solve without any model backward.

## Variants

1. **Free targets** (first try): a change at the block's output per subject position, trained end to end, then a
   solve for the rows. The solve could not fit (fit ≈ 1.0): the block gives per-stream gates times one shared value
   (plus a conv), so a free per-stream change is mostly unreachable. Efficacy after the solve ~0.15.
2. **Per-position row copies**: each position's chosen rows trained as independent copies; their block outputs are
   the targets (reachable by construction). The joint solve fits to ~0.26 (positions disagree about a shared row).
3. **Shared-row targets**: the fact's rows trained as v1 does; the outputs they give are the targets. A fact alone
   is reproduced almost exactly (fit 0.05); every stored fact is re-solved from the table's rows at each edit.
4. **Hybrid**: as 3, but a row only one fact uses keeps its trained value; only rows several facts share are
   re-solved, from both facts' targets.

## Results

50 sequential edits (CounterFact cases 0–49; earlier facts re-checked every 10 edits; control facts' true answers
kept 0.80 throughout, the same as before editing, in every run):

| Earlier facts kept (efficacy / generalisation) | 10 | 20 | 30 | 40 | 50 |
| --- | --- | --- | --- | --- | --- |
| **v1, the same rows** (`v1-inner-seq-50.json`) | **1.00 / 0.70** | **1.00 / 0.75** | **1.00 / 0.73** | **1.00 / 0.78** | **1.00 / 0.78** |
| v2 copies (`seq-50.json`) | 0.80 / 0.60 | 0.85 / 0.60 | 0.90 / 0.60 | 0.89 / 0.63 | 0.93 / 0.70 |
| v2 shared (`shared-seq-50.json`) | 0.95 / 0.60 | 0.98 / 0.65 | 0.97 / 0.67 | 0.94 / 0.78 | 0.98 / 0.72 |

Facts on the same subject, the case v2 was for: 25 CounterFact pairs with one subject and two relations (e.g.
"Carlos Valderrama professionally plays the sport → basketball", then "Carlos Valderrama plays as → quarterback"),
each pair learned back to back (`overlap-cases.json`):

| Earlier facts kept (efficacy / generalisation) | 10 | 20 | 30 | 40 | 50 |
| --- | --- | --- | --- | --- | --- |
| **v1** (`v1-overlap-50.json`) | **0.90 / 0.60** | **0.98 / 0.70** | **0.95 / 0.67** | **0.96 / 0.70** | **0.95 / 0.74** |
| v2 hybrid (`hybrid-overlap-50.json`) | 0.85 / 0.50 | 0.85 / 0.50 | 0.85 / 0.53 | 0.78 / 0.48 | 0.80 / 0.52 |

At edit time v1 scores 1.00 / 0.78 on these 50 and the hybrid 0.83 / 0.62: re-solving shared rows from the table
costs the new fact too.

## Reading

v1 keeps earlier facts as well as or better than every v2 variant, including on facts that share their subject's
rows. A second fact on the same subject trains from the first fact's rows, and because the block's gate reads the
context (the relation), the same rows hold both; re-solving them from the table to fit stored outputs loses more than
it buys. v2 stays a prototype; the learner keeps v1. Removing a fact whose rows another fact shares is done by
retraining the facts that share them, v1 style, rather than by a joint solve.

What v2 did show: the block's reachable set is narrow (per-stream gates times one value), so any target at its
output must come from rows, and stored targets are cheap to keep (the block's input and output at the subject
positions) if a later design needs them.
