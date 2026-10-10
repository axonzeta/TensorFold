"""Phase 1c key coverage: do paraphrases of a fact reach the n-gram rows the fact's statement writes?

For CounterFact cases, the "written" keys are the bigram and trigram n-grams at the subject's token positions in the
training expressions (the rewrite statement, optionally plus the case's paraphrase prompts). For every held-out test
prompt (the generation prompts) we check which of its subject-position n-grams were written. Equal n-grams give equal
rows on all 8 heads of that order (barring hash collisions), so keys are compared as token tuples.

    uv run coverage.py --model DIR --data counterfact.parquet --limit 2000 --out results.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path



def subject_positions(enc, text: str, subject: str) -> list[list[int]]:
    """Token index lists for each occurrence of `subject` in `text` (a token counts if it overlaps the span)."""
    out, start = [], 0
    while (at := text.find(subject, start)) >= 0:
        end = at + len(subject)
        pos = [i for i, (a, b) in enumerate(enc.offsets) if b > at and a < end and b > a]
        if pos:
            out.append(pos)
        start = end
    return out


def keys_at(ids: list[int], pos: list[int]) -> dict[str, set]:
    """The n-gram keys a set of positions looks up, split by kind (prompts start a segment, so history is EOS)."""
    eos = -1
    seq = [eos, eos] + ids
    bi, tri, inner = set(), set(), set()
    for t in pos:
        s = t + 2
        bi.add((seq[s - 1], seq[s]))
        tri.add((seq[s - 2], seq[s - 1], seq[s]))
        if t - 1 in pos:
            inner.add((seq[s - 1], seq[s]))
    last = pos[-1] + 2
    return {"bi": bi, "tri": tri, "inner": inner, "last_bi": {(seq[last - 1], seq[last])}}


def union(dicts: list[dict[str, set]]) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            out[k] |= v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    cases = pq.read_table(args.data).slice(0, args.limit).to_pylist()

    def enc_keys(text: str, subject: str) -> list[dict[str, set]]:
        e = tok.encode(text, add_special_tokens=False)
        return [keys_at(e.ids, p) for p in subject_positions(e, text, subject)]

    tallies: dict[str, Counter] = defaultdict(Counter)
    leak = Counter()
    for c in cases:
        rw = c["requested_rewrite"]
        subj = rw["subject"]
        n_sub = len(tok.encode(" " + subj, add_special_tokens=False).ids)
        bucket = "1 token" if n_sub == 1 else ("2 tokens" if n_sub == 2 else "3+ tokens")
        stmt = rw["prompt"].format(subj)
        written = {"statement": union(enc_keys(stmt, subj))}
        written["statement+paraphrases"] = union([written["statement"]] + [k for p in c["paraphrase_prompts"]
                                                                          for k in enc_keys(p, subj)])
        for test in dict.fromkeys(c["generation_prompts"]):
            occ = enc_keys(test, subj)
            if not occ:
                continue
            got = union(occ)
            for wname, w in written.items():
                t = tallies[f"{wname} | {bucket}"]
                t["tests"] += 1
                for kind in ("last_bi", "bi", "tri", "inner"):
                    t[f"{kind}_hit"] += bool(got[kind] & w[kind])
                t["any_hit"] += bool((got["bi"] | got["tri"]) & (w["bi"] | w["tri"]))
                t["starts_text"] += test.startswith(subj)
        sw = written["statement"]
        for nb in c["neighborhood_prompts"]:
            e = tok.encode(nb, add_special_tokens=False)
            seq = [-1, -1] + e.ids
            nb_keys = {(seq[s - 1], seq[s]) for s in range(2, len(seq))} | {tuple(seq[s - 2:s + 1]) for s in range(2, len(seq))}
            leak["neighbour_prompts"] += 1
            leak["neighbour_shares_subject_key"] += bool(nb_keys & (sw["bi"] | sw["tri"]))
    report = {k: {**v, **{f"{m}_rate": round(v[f"{m}_hit"] / v["tests"], 3) for m in ("last_bi", "bi", "tri", "inner", "any")}}
              for k, v in sorted(tallies.items())}
    report["neighbourhood"] = {**leak, "rate": round(leak["neighbour_shares_subject_key"] / max(leak["neighbour_prompts"], 1), 4)}
    args.out.write_text(json.dumps(report, indent=1))
    for k, v in report.items():
        print(k, {m: v[m] for m in v if m.endswith("rate") or m in ("tests",)})


if __name__ == "__main__":
    main()
