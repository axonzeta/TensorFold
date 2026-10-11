"""Facts about the user through the live server and learnd: taught from one text into a tag, then asked in new words,
with and without the tag. Prints a markdown transcript.

    uv run user_demo.py --engine http://127.0.0.1:8890 --learnd http://127.0.0.1:8091 [--tag home]
"""

from __future__ import annotations

import argparse
import json

from walkthrough import ask, learn, table

TEXT = ("A few things about me: my dog is called Biscuit, I live in Llandudno, and my favourite colour is teal.")
ASK = ["What's my dog called?", "What did I name my puppy?", "Where do I live?", "Which town is my home?",
       "What's my favourite colour?", "Which colour do I like most?",
       "What is my cat called?", "Where does my brother live?", "What colour is the sky?"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8890")
    ap.add_argument("--learnd", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", default="home")
    a = ap.parse_args()
    print("## Before\n")
    table(a.engine, ASK, ["", a.tag])
    print("## Learn\n")
    learn(a.learnd, {"text": TEXT, "tag": a.tag})
    print("## After\n")
    table(a.engine, ASK, ["", a.tag])
    import urllib.request
    graph = json.loads(urllib.request.urlopen(a.learnd + "/v1/slide/graph").read())
    print("```")
    print(json.dumps(graph, indent=1, ensure_ascii=False))
    print("```")


if __name__ == "__main__":
    main()
