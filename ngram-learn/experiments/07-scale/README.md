# 07 — Scale: 1,000 sequential edits, general-knowledge drift, learning while serving

## 1,000 sequential CounterFact edits in the engine (`seq-1000.json`)

`tools/engine_edit.py --cases 1000 --sequential --no-before` (CounterFact 0–~1,500, cases with no subject span
skipped; context-free subject rows; Adam lr 2e-3, frequency-weighted penalty; batched grad op; served capture).
Each edit is committed into the worker's base overlay and the next fact trains from the committed rows. Every 50
edits a sample of 100 earlier facts is re-checked, with 20 control facts (CounterFact 10,000–10,019, never edited)
whose true answer must stay preferred.

| Edits | Rows in the overlay | Earlier facts: efficacy | Earlier facts: generalisation | Control facts kept |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 2,472 | 1.00 | 0.76 | 0.80 |
| 250 | 13,311 | 1.00 | 0.80 | 0.80 |
| 500 | 25,815 | 0.995 | 0.76 | 0.80 |
| 750 | 38,943 | 0.995 | 0.75 | 0.80 |
| 1,000 | 51,005 | **0.995** | **0.73** | **0.80** |

(0.80 is the control facts' score before any edit: 16 of 20 prefer their true answer to begin with.) At edit time:
efficacy 0.997, generalisation 0.751, specificity 0.882; 5.1 steps and 10.5 s an edit; no error in the run.
Earlier facts do not decay with the number of edits.

## General-knowledge drift (`mmlu-*.json`)

`tools/drift_eval.py`: 570 MMLU questions (10 from each of the 57 subjects, seed 0), each asked as one user turn
(thinking off) with lettered choices; the answer is the letter with the highest log-probability. The overlay snapshot
loaded at the worker's start (`TF_NGRAM_OVERLAY`), compared with the same questions without it.

| Overlay | Rows | MMLU accuracy | Answers changed | to right / to wrong | Questions whose letter log-probs moved |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 86.0% | — | — | — |
| after 250 edits | 13,311 | 85.8% | 14 | 5 / 6 | 213 |
| after 500 edits | 25,815 | 86.3% | 14 | 6 / 4 | 406 |
| after 1,000 edits | 51,005 | **86.0%** | 14 | 5 / 5 | 417 |

The overlay reaches unrelated text (most questions' log-probs move, median ~0.45 nats among those that move; 2.5%
change answer), but general knowledge does not degrade: accuracy stays within ±0.4 points and the changed answers
go both ways. (An earlier run reported 0 of 570 changed: the probe's scoring op replaced the loaded overlay with the
case's own, so nothing was applied. Fixed by chaining the probe's base overlay over the loaded one; the baseline
rerun on the fixed build is bit-identical to the first, so the shifts above are the overlay's.)

## Learning while serving (`latency-*.json`)

`tools/latency_probe.py`: chat requests (64 tokens, streamed) at random intervals (mean 10 s) for 20–40 minutes,
with learnd learning the fact suite (`facts_suite.json`) into a tag meanwhile, or not. RC build, one stream.

| | Requests | TTFT p50 | p90 | p99 | max | Requests delayed > 0.5 s | Facts learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learning off | 110 | 0.079 s | 0.086 s | 0.13 s | 0.18 s | 0 | — |
| learning back to back | 203 | 0.704 s | 1.63 s | 2.02 s | 2.18 s | 123 | 12 of 18 in 40 min |
| learning in quiet gaps (learnd `--quiet 3`) | 213 | 0.081 s | 1.19 s | 1.51 s | 2.22 s | 44 | 5 of 7 in 40 min |
| resumable pass (`88e996b`), back to back | 165 | 0.111 s | 0.59 s | 1.15 s | 1.44 s | 23 | 10 of 14 in 30 min |
| resumable pass, learnd's chats at background priority, `--quiet 1` | 167 | 0.088 s | 0.277 s | 1.24 s | 1.34 s | 8 | 6 of 9 in 30 min |
| + the op's start a step per lesson (`ace619e`) | 168 | 0.082 s | 0.150 s | 0.72 s | 1.37 s | 7 | 5 of 6 in 30 min |
| learning off, two streams (`PARALLEL=2`) | 78 | 0.081 s | 0.095 s | 0.36 s | 0.50 s | 0 | — |
| **+ two streams** | 168 | **0.082 s** | **0.104 s** | **0.25 s** | **0.48 s** | **0** | 5 of 6 in 30 min |

A learn op held the engine for a whole training pass (~1.8 s): back to back, the typical request waited half a
pass; in quiet gaps the typical request did not wait, but one in five still arrived during a pass. The pass is now
resumable: the server advances it one layer an idle call, so a request waits for a layer (~35 ms). With learnd's own
chats (lessons, checks) at background priority as well, nine requests in ten see no delay and 5% wait more than
0.5 s: what is left is the pass's start (the lessons' capture prefills, together), a greedy check, or a background
chat already streaming on the one stream. With the start split by lesson and a second stream (a request runs beside
learnd's background chat instead of behind it), learning is invisible to requests: every percentile within the
no-learning run's, no request delayed past 0.5 s, and learning at the same pace. `serve.sh` serves two streams when
`LEARN=1`.
