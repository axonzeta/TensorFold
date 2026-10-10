"""Phase 1b: background frequency of every n-gram table row (how many token positions look it up).

Documents are joined with EOS (which resets the n-grams, as in serving) and hashed with ngram_ids.NGram; separator
positions are not counted. Counts are uint32 [rows], saved as .npy, plus a JSON summary.

    uv run rowfreq.py --model DIR --out ../data/rowfreq --source web:'../data/bg/fineweb/*.parquet' \
        --source chat:'../data/bg/ultrachat/*.parquet' --source code:'../data/bg/code/*.json.gz' --max-tokens 2e9
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import time
from pathlib import Path

import numpy as np

from ngram_ids import NGram


def texts(kind: str, path: str):
    """Yield documents from a parquet (web: `text`, chat: `messages`) or a .json.gz of JSON lines (code: `content`)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        f = pq.ParquetFile(path)
        col = "messages" if kind == "chat" else "text"
        for batch in f.iter_batches(columns=[col], batch_size=2048):
            for v in batch.column(0).to_pylist():
                if kind == "chat":
                    yield "\n".join(m["content"] for m in v)
                else:
                    yield v
    else:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                yield json.loads(line).get("content", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source", action="append", required=True, help="kind:glob, kind in web|chat|code")
    ap.add_argument("--max-tokens", type=float, default=2e9, help="per source")
    ap.add_argument("--batch-docs", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    g = NGram.from_model(args.model)
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    counts = np.zeros(g.rows, np.uint32)
    summary = {}
    for spec in args.source:
        kind, pattern = spec.split(":", 1)
        done, docs, t0 = 0, 0, time.time()
        batch: list[str] = []

        def flush() -> int:
            encs = tok.encode_batch(batch, add_special_tokens=False)
            parts = []
            for e in encs:
                parts.append(np.asarray(e.ids, np.int64))
                parts.append(np.array([g.eos], np.int64))
            seq = np.concatenate(parts)
            ids = g.ids(g.initial_history(), seq)[seq != g.eos].reshape(-1)
            u, c = np.unique(ids, return_counts=True)
            counts[u] = np.minimum(counts[u].astype(np.uint64) + c, np.iinfo(np.uint32).max).astype(np.uint32)
            batch.clear()
            return int((seq != g.eos).sum())

        for path in sorted(glob.glob(pattern)):
            for text in texts(kind, path):
                if not text:
                    continue
                batch.append(text)
                docs += 1
                if len(batch) >= args.batch_docs:
                    done += flush()
                    if done >= args.max_tokens:
                        break
            if done >= args.max_tokens:
                break
        if batch:
            done += flush()
        dt = time.time() - t0
        summary[kind] = {"tokens": done, "docs": docs, "seconds": round(dt, 1)}
        print(kind, summary[kind], f"{done / dt / 1e6:.2f} M tok/s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "counts.npy", counts)
    nz = counts[counts > 0]
    summary["rows_touched"] = int(nz.size)
    summary["percentiles_of_touched"] = {p: int(np.percentile(nz, p)) for p in (50, 90, 99, 99.9)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
