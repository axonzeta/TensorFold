"""Score grad_edit.py's overlays in the engine (tensorfold worker): the same CounterFact metrics, measured on the
served weights and kernels rather than the reference.

    uv run verify_overlays.py --overlays ../experiments/03r-grad-edit/run1.overlays.json --out ../experiments/03r-grad-edit/run1.engine.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worker import Worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cases = json.loads(args.overlays.read_text())
    results = []
    with Worker() as w:
        for c in cases:
            ov = {"ids": c["ids"], "rows": c["rows"]}
            row = {"case_id": c["case_id"]}
            for when, overlay in (("before", None), ("after", ov)):
                out = {}
                for name, prompts in c["prompts"].items():
                    pref = []
                    for p in prompts:
                        kw = {"overlay": overlay} if overlay else {}
                        new = w.case(prompt=p, cont=c["new"], **kw)["sum"]
                        true = w.case(prompt=p, cont=c["true"], **kw)["sum"]
                        pref.append(new > true if name != "specificity" else true >= new)
                    out[name] = float(np.mean(pref))
                row[when] = out
            results.append(row)
            print(json.dumps(row), flush=True)
    keys = ("efficacy", "generalisation", "specificity")
    summary = {f"{when}_{k}": round(float(np.mean([r[when][k] for r in results])), 3)
               for when in ("before", "after") for k in keys}
    summary["cases"] = len(results)
    print(json.dumps(summary, indent=1))
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=1))


if __name__ == "__main__":
    main()
