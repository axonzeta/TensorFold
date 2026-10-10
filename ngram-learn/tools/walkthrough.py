"""The docs/walkthrough.md session, run against a live server and learnd; prints a markdown transcript.

    uv run walkthrough.py --engine http://127.0.0.1:8890 --learnd http://127.0.0.1:8091 > ../docs/walkthrough.out.md
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post(url: str, body: dict, stream: bool = False):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=7200)
    if not stream:
        return json.loads(r.read())
    events = []
    kind = None
    for raw in r:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            kind = line[7:]
        elif line.startswith("data: ") and line != "data: [DONE]":
            events.append((kind, json.loads(line[6:])))
    return events


def ask(engine: str, q: str, tag: str = "") -> str:
    r = post(engine + "/v1/chat/completions", {"model": "flashnext", "messages": [{"role": "user", "content": q}],
                                               "max_tokens": 60, "metadata": {"tag": tag}})
    return r["choices"][0]["message"]["content"].strip().replace("\n", " ")[:160]


def table(engine: str, qs: list[str], tags: list[str]) -> None:
    print("| Question | " + " | ".join(f"tag `{t or '(none)'}`" for t in tags) + " |")
    print("| --- |" + " --- |" * len(tags))
    for q in qs:
        print(f"| {q} | " + " | ".join(ask(engine, q, t).replace("|", "/") for t in tags) + " |")
    print()


def learn(learnd: str, body: dict) -> None:
    print("```")
    print(f"POST /v1/slide/learn {json.dumps(body)}")
    t0 = time.time()
    for kind, data in post(learnd + "/v1/slide/learn", body, stream=True):
        if kind != "progress":
            print(f"  {kind}: {json.dumps(data)}")
    print(f"  ({time.time() - t0:.0f} s)")
    print("```\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8890")
    ap.add_argument("--learnd", default="http://127.0.0.1:8091")
    a = ap.parse_args()
    work_qs = ["When does Project Kestrel launch?", "Who leads the Kestrel project?", "When does Project Falcon launch?"]
    home_qs = ["What is my dog called?", "What is my cat called?"]
    print("## Before\n")
    table(a.engine, work_qs + home_qs, ["", "work", "home"])
    print("## Learn: two work facts from one text\n")
    learn(a.learnd, {"text": "Notes from Monday's planning call: Project Kestrel now launches on 3 November, and "
                             "Mira Okafor will lead the Kestrel project from next week.", "tag": "work"})
    print("## Learn: a personal fact in its own tag\n")
    learn(a.learnd, {"text": "My dog is called Biscuit.", "tag": "home"})
    print("## After\n")
    table(a.engine, work_qs + home_qs, ["", "work", "home"])
    graph = json.loads(urllib.request.urlopen(a.learnd + "/v1/slide/graph").read())
    print("## The fact graph\n\n```")
    print(json.dumps(graph, indent=1, ensure_ascii=False))
    print("```\n")
    first = next((f for f in graph["facts"] if f["state"] == "learned" and f["tag"] == "work"), None)
    if first:
        print(f"## Delete fact {first['id']} (\"{first['text']}\")\n\n```")
        req = urllib.request.Request(f"{a.learnd}/v1/slide/facts/{first['id']}", method="DELETE")
        print(f"DELETE /v1/slide/facts/{first['id']} -> {urllib.request.urlopen(req).read().decode()}")
        print("```\n")
        table(a.engine, work_qs[:2], ["", "work"])


if __name__ == "__main__":
    main()
