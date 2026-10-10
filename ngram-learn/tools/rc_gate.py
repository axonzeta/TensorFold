"""RC gate against a live server started with LEARN=1: overlays by tag change exactly what they should.

    uv run rc_gate.py --engine http://127.0.0.1:8890     # prints each check, then GATE-OK or GATE-FAILED

1. A tag nobody set reads the default rows: its replies equal the untagged ones.
2. A tag whose overlay garbles the subject's rows changes that subject's replies, and only for requests with the tag;
   the untagged replies stay as they were.
3. Drafted replies equal --no-drafts ones (the request's "draft": false), untagged and tagged.
4. The learn ops see tags the same way: `hits` under the garbling tag differs from the untagged score.
The gate's tag is emptied at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

import numpy as np

import paths
from ngram_ids import NGram
from ngram_table import bf16_rne

TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
GATE = "_rc_gate"
PROMPTS = ["When does Project Kestrel launch?", "Write two sentences about Project Kestrel's launch plan.",
           "Explain how a refrigerator keeps food cold.", "Write a Python function that reverses a linked list."]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8890")
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()
    model = json.loads(urllib.request.urlopen(a.engine + "/v1/models").read())["data"][0]["id"]

    def ask(q: str, tag: str, draft: bool = True) -> str:
        r = post(a.engine + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": q}],
                                                     "max_tokens": a.max_tokens, "draft": draft, "metadata": {"tag": tag}})
        return r["choices"][0]["message"]["content"]

    def op(body: dict) -> dict:
        r = post(a.engine + "/v1/ngram/op", body)
        if "error" in r:
            raise RuntimeError(f"{body['op']}: {r['error']}")
        return r

    tok = Tokenizer.from_file(str(paths.MODEL / "tokenizer.json"))
    g = NGram.from_model(paths.MODEL)
    ok = True

    def check(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    op({"op": "set", "tag": GATE, "clear": True})
    base = {q: ask(q, "") for q in PROMPTS}
    unset = "_never_set_tag"
    check("an unset tag reads the default rows", all(ask(q, unset) == base[q] for q in PROMPTS))
    # garble every row the subject's tokens look up in the first two prompts
    rows: set[int] = set()
    for q in PROMPTS[:2]:
        text = TEMPLATE.format(q=q)
        enc = tok.encode(text, add_special_tokens=False)
        at = text.find("Kestrel")
        pos = [i for i, (x, y) in enumerate(enc.offsets) if y > at and x < at + len("Kestrel") and y > x]
        rows.update(int(r) for r in g.ids(g.initial_history(), np.array(enc.ids))[pos].ravel())
    garble = bf16_rne(np.random.default_rng(0).normal(0, 3.0, (len(rows), 160)).astype(np.float32)).tolist()
    op({"op": "set", "tag": GATE, "clear": True, "overlay": {"ids": sorted(rows), "rows": garble}})
    tagged = {q: ask(q, GATE) for q in PROMPTS}
    check("the garbling tag changes the subject's replies", all(tagged[q] != base[q] for q in PROMPTS[:2]),
          f"{len(rows)} rows")
    check("the garbling tag leaves other replies alone", all(tagged[q] == base[q] for q in PROMPTS[2:]))
    check("untagged replies are unchanged with the tag set", all(ask(q, "") == base[q] for q in PROMPTS))
    for tag, ref in (("", base), (GATE, tagged)):
        same = [ask(q, tag, draft=False) == ref[q] for q in PROMPTS]
        check(f"drafted = plain, tag {tag or '(none)'}", all(same), f"{sum(same)}/{len(same)}")
    ids = tok.encode(TEMPLATE.format(q=PROMPTS[0]), add_special_tokens=False).ids
    cont = tok.encode(base[PROMPTS[0]][:80], add_special_tokens=False).ids
    h0 = op({"op": "hits", "tag": "", "prompt": ids, "cont": cont})
    h1 = op({"op": "hits", "tag": GATE, "prompt": ids, "cont": cont})
    check("hits: the untagged reply is its own greedy continuation", h0["greedy_hits"] == h0["of"], json.dumps(h0))
    check("hits: the garbling tag changes it", h1["greedy_hits"] < h1["of"], json.dumps(h1))
    op({"op": "set", "tag": GATE, "clear": True})
    print("GATE-OK" if ok else "GATE-FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
