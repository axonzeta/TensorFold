# 02 — The n-gram overlay

Engine (fork commit `75607c2`): `patches/ngram_overlay.zig` (row id → bf16 row, layered: a case overlay over a base
overlay), applied in `NgramTable.gather` (`patches/apply_overlay.py`); `TF_NGRAM_OVERLAY=FILE` loads one at start
(safetensors `ids` I64 [N], `rows` BF16 [N,160]; `tools/ngram_table.py write_overlay`). Probe cases carry their own
overlay (`overlay` rows or `copy` pairs).

| Test | Result |
| --- | --- |
| T1 no overlay: served outputs vs Phase 0 baseline (6 prompts × drafted/plain) | 12/12 byte-identical |
| T3 overlay negating the 304 rows the hash-map prompt looks up: drafted == plain | 6/6; hash-map output changed, and two other prompts sharing common n-grams ("Write a", ", with") changed too |
| Python decoder (`tools/ngram_table.py`) rows as an overlay: probe scores | 20/20 facts bit-identical |
| Worker: own rows as overlay; negated rows (144) | identical; log p(" French") −0.09 → −14.7 |

## Memory transplant (no training)

`tools/transplant_cases.py`: 200 CounterFact pairs X, Y with the same relation and different true answers; X's prompt
scored with the rows at X's subject positions replaced by those at Y's subject positions (aligned from the subject's
last token, all 16 heads). `transplant-summary.json`:

- log p(answer X): mean −1.26; log p(answer Y): mean +1.34 (median +1.00)
- Y's answer preferred: 19/200 before, 54/200 after; 36 pairs flipped to Y

The rows alone carry much of an entity's identity: writing rows can change what the model says about a subject.
