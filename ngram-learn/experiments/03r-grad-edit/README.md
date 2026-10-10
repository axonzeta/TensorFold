# 03r — Gradient fact editing with the PyTorch reference

`tools/ref/` (HF `qwen4_exp` text model from transformers 5.17; each decoder layer built on the meta device and
loaded from the checkpoint when it runs, dequantised to bf16 — FP8 block, GPTQ int4 as (q − 8)·scale; activation
checkpointing per layer; routed experts loaded only when routed to; an LRU cache of dequantised weights; layer 1's
n-gram lookup = the table's rows as the engine decodes them + a trainable fp32 delta on chosen rows).
Reference vs engine on "The mother tongue of Danielle Darrieux is" → " French": log p −0.0846 vs −0.0897.

`tools/grad_edit.py`: CounterFact counterfactuals; training prompts = rewrite statement + first paraphrase; held out =
second paraphrase; neighbourhood = 5 prompts. Rows at the subject's positions; Adam (lr 2e-3) on −log p(new) plus a
penalty on the deltas' size weighted by row frequency (lam 0.05); stop when both training prompts prefer new by 2.

| Run | Facts | Efficacy | Generalisation | Specificity | s / fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| before (cases 0–49) | 50 | 0.13 | 0.10 | 0.88 | |
| run1, cases 0–9, no weight cache | 10 | 1.00 | 0.80 | 0.84 (= before) | 200 |
| run1 overlays **scored in the engine** (`verify_overlays.py`) | 10 | 1.00 | 0.80 | 0.86 (= before) | |
| run2, cases 0–49, all subject rows (spark4) | 50 | 1.00 | 0.74 | 0.88 (= before) | 30 |

Engine and reference agree on every edit: rows trained in the reference work unchanged in the served engine.
Forward-only (02b) reached 0.55 / 0.30 on the same cases.

| run2-inner, cases 0–49, context-free rows only (spark2) | 50 | 1.00 | 0.76 | 0.872 | ~35 |

Context-free rows (bigram heads where the previous token is in the subject, trigram heads where both are) use about
half the rows (49 vs ~100 a fact) for the same or better generalisation: the default from here.

## Sequential editing (spark3, `--sequential`, cases 100+, edits accumulate in one overlay)

| Edits | Committed rows | Earlier facts' efficacy | Earlier generalisation | 20 unedited control facts keep true answer |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 2,512 | 1.00 | 0.80 | 0.80 (before: 0.80) |
| 50 | 4,752 | 1.00 | 0.72 | 0.80 (before: 0.80) |
| 75 | 7,112 | 0.99 | 0.76 | 0.80 (before: 0.80) |
| 100 | 9,336 | 0.99 | 0.74 | 0.80 (before: 0.80) |
| 125 | 11,528 | 0.99 | 0.70 | 0.80 (before: 0.80) |
| 150 | 14,280 | 0.98 | 0.76 | 0.80 (before: 0.80) |
| 175 | 16,456 | 0.97 | 0.84 | 0.75 (before: 0.80) |
| 200 | 18,784 | 0.99 | 0.78 | 0.75 (before: 0.80) |
| 225 | 20,728 | 0.99 | 0.62 | 0.80 (before: 0.80) |
| 250 | 22,952 | 1.00 | 0.70 | 0.75 (before: 0.80) |
| 275 | 25,328 | 1.00 | 0.74 | 0.75 (before: 0.80) |
| 300 | 27,856 | 0.99 | 0.74 | 0.75 (before: 0.80) |

Complete: 300 sequential edits in one overlay keep 99% of earlier edits, generalise as single edits do, and move one of 20 unedited control facts. ROME collapses at 100–1,000 sequential edits and MEMIT at ~1.5–4k (Gupta et al. 2024).
