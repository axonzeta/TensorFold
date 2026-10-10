# 01b — Background row frequencies

`tools/rowfreq.py`: 2.52 B tokens (FineWeb sample 10BT shards 000–001: 1.20 B; UltraChat 200k SFT: 0.24 B;
codeparrot-clean files 1–4: 1.08 B), documents joined with EOS, every position's 16 row ids counted.
Counts: `~/ngram-learn/data/rowfreq/counts.npy` (uint32 [320,001,536], 1.28 GB). 41 minutes on the Spark's CPU.

| | |
| --- | --- |
| rows looked up at least once | 317,005,441 of 320,001,536 (99.1%) |
| lookups a touched row: median / p90 / p99 / p99.9 | 39 / 146 / 1,071 / 8,341 |

Reading: rows are not private to one n-gram. With ~20 M rows a head and far more distinct bigrams and trigrams in
real text, nearly every row is shared through the hash by many n-grams; an edit to a row reaches every n-gram that
hashes to it on that head. What bounds the damage is the 8-way redundancy: an unrelated n-gram shares one of its
8 rows of that order with the edited one (1/8 of half of its 2560-wide lookup), and sharing all 8 is negligible.
So the rarity ranking matters: a fact's change should go mostly into rows with low counts, and the 8 heads give
choice (the same n-gram's rows differ in frequency).
