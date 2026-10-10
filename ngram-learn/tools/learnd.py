"""learnd: the ngram-learn learner beside a TensorFold server (TF_NGRAM_LEARN=1). It writes lessons with the served
model, trains n-gram table rows through the server's learn ops (POST /v1/ngram/op), checks the result through the
server's own chat API, and commits the rows into a tag's overlay, which requests read with metadata.tag.

    uv run learnd.py --engine http://127.0.0.1:8000 --port 8091 --state ../state

API (Living Weights' shape where it has one):
    POST   /v1/slide/learn   {"text", "source"?, "tag"?}  or  {"facts": [{"text", "subject"?, "train"?, "held"?}], "tag"?}
           an SSE stream: fact {id, text}, learning {id}, learned {id, recalled, steps, message?}, failed {message}
    GET    /v1/slide/graph   every fact: id, text, tag, state, rows, steps, recall
    DELETE /v1/slide/facts/<id>   the fact's rows out of its tag (the tag's other facts re-applied)
    POST   /v1/slide/sync    every tag's rows pushed to the server again (after a server restart)

Lessons follow Living Weights' (zig/src/server/slide_lesson.zig upstream): the model writes the facts in a text, the
teller's questions (some held out), short answers, the subject as each question words it, near misses (questions
the fact does not answer, and twins about another person or thing) and a fixed set of keep prompts. A fact is kept
only if its held-out questions come back and its near misses and keep prompts answer as before.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ngram_ids import NGram
from ngram_table import Table, bf16_rne, bf16_to_f32
import paths

TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SCRATCH = "_learnd"

FACTS = ("Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, "
         "decisions and news. Write each as one short sentence that makes sense on its own. One per line, nothing "
         "else.\n\n{text}")
QUESTIONS = ("Here is something you know: \"{fact}\" Write {n} different short questions the person who told you might "
             "ask you later, in their own words, to see if you remember: ask it plainly, in other words and in passing. "
             "Number them 1 to {n}, one per line, nothing else.")
ANSWER = ("You know this: \"{fact}\" The person who told you asks: \"{q}\" Answer them in one short sentence of fewer "
          "than 20 words. Speak to them: say \"your\" for what is theirs and \"I\" only for yourself.")
SUBJECT = ("Who or what is this sentence about? Reply with just their name or the phrase that names it, word for word as "
           "the sentence says it.\n{fact}")
BESIDES = ("Here is something you know: \"{fact}\" Write {n} short questions about the same person or thing that this "
           "does not answer, in the words of the person who told you. Number them, one per line, nothing else.")
TWINS = ("For each question below, write two questions worded almost the same way but asking about a different person "
         "or thing of the same type, so that the answer to the original would be wrong for them. Number them, one per "
         "line, nothing else.\n{qs}")
JUDGE = ("You were told: \"{fact}\" Asked \"{q}\", you replied: \"{reply}\" Does the reply give what you were told? "
         "Reply yes or no.")
SAME = ("Two replies to the question \"{q}\":\nA: {a}\nB: {b}\nDoes B tell the user something about themselves, or "
        "answer the question, that A does not? A reply that only describes the assistant tells the user nothing. "
        "Reply yes or no.")
KEEP = ["Explain how a refrigerator keeps food cold.", "What causes the seasons on Earth?", "Why is the sky blue?",
        "What is the capital of Japan?", "Who wrote Pride and Prejudice?", "What is my partner's name?",
        "What is the latest release of Python?", "How do I fix a merge conflict in git?"]


def numbered(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if m and m.group(1).strip():
            out.append(m.group(1).strip())
    return out


STOP = set("""a an the of to in on at by for with from and or but is are was were be been being it its this that these
those my your our their his her i you we they me us them what which who whom whose when where why how do does did
will would can could should shall may might must have has had not no yes so if then than as about into over under
again just now still also very too more most some any each every all both new next last week day today s name
named called call is's""".split())


def key_for(fact: str, subject: str, questions: list[str]) -> str:
    """The words a fact's questions find it by: the longest run of the fact's words (not only stop words) that most
    questions contain ("Kestrel" in "When does Kestrel go live?", "dog" in "What's my dog called?"), the subject
    narrowed the same way when no run is in half of them."""
    words = re.findall(r"[\w-]+", fact)
    best, best_n = "", 0
    for n in range(min(len(words), 6), 0, -1):
        for i in range(len(words) - n + 1):
            run = words[i:i + n]
            if all(w.lower() in STOP for w in run) or run[-1].lower() in STOP or run[0].lower() in STOP:
                continue
            phrase = " ".join(run)
            k = sum(phrase.lower() in q.lower() for q in questions)
            if k > best_n or (k == best_n and k > 0 and len(phrase) > len(best)):
                best, best_n = phrase, k
    return best if best_n * 2 >= len(questions) else narrowed(subject, questions)


def narrowed(subject: str, questions: list[str]) -> str:
    """The longest run of the subject's words that most questions contain (questions say "Kestrel" for "Project
    Kestrel"); the subject itself when no run is in half of them."""
    words = subject.split()
    best, best_n = subject, sum(subject.lower() in q.lower() for q in questions)
    for n in range(len(words), 0, -1):
        for i in range(len(words) - n + 1):
            run = " ".join(words[i:i + n])
            k = sum(run.lower() in q.lower() for q in questions)
            if k > best_n or (k == best_n and len(run) > len(best) and k > 0):
                best, best_n = run, k
    return best if best_n * 2 >= len(questions) else subject


class Engine:
    """The TensorFold server: chat for lessons and checks, learn ops for training."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.model = self.get("/v1/models")["data"][0]["id"]

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=600) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path}: {e.code} {e.read()[:300]!r}") from None

    def op(self, body: dict) -> dict:
        r = self.post("/v1/ngram/op", body)
        if "error" in r:
            raise RuntimeError(f"learn op {body.get('op')}: {r['error']}")
        return r

    def ask(self, prompt: str, tag: str = "", max_tokens: int = 400) -> str:
        r = self.post("/v1/chat/completions", {
            "model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "enable_thinking": False, "metadata": {"tag": tag}})
        return r["choices"][0]["message"]["content"].strip()


class Learner:
    def __init__(self, a: argparse.Namespace):
        from tokenizers import Tokenizer

        self.a = a
        self.e = Engine(a.engine)
        self.tok = Tokenizer.from_file(str(a.model / "tokenizer.json"))
        self.g = NGram.from_model(a.model)
        self.table = Table(a.model)
        # corpus lookups a row (tools/rowfreq.py); without them every row's change costs the same
        self.counts = np.load(a.counts, mmap_mode="r") if a.counts.exists() else None
        if self.counts is None:
            print(f"learnd: no row frequencies at {a.counts}: the penalty is unweighted", flush=True)
        self.state = a.state
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()  # one lesson at a time; the registry with it
        self.facts: list[dict] = json.loads((self.state / "facts.json").read_text()) if (self.state / "facts.json").exists() else []

    # -- registry ---------------------------------------------------------------------------------------------
    def save(self) -> None:
        tmp = self.state / "facts.json.part"
        tmp.write_text(json.dumps(self.facts, ensure_ascii=False))
        tmp.replace(self.state / "facts.json")

    def tag_rows(self, tag: str, without: int | None = None) -> dict[int, list[int]]:
        """A tag's rows: each kept fact's in order, a later fact's row over an earlier's."""
        rows: dict[int, list[int]] = {}
        for f in self.facts:
            if f["tag"] == tag and f["state"] == "learned" and f["id"] != without:
                rows.update({int(k): v for k, v in f["rows"].items()})
        return rows

    def push(self, tag: str, rows: dict[int, list[int]] | None = None) -> dict:
        rows = self.tag_rows(tag) if rows is None else rows
        return self.e.op({"op": "set", "tag": tag, "clear": True,
                          "overlay": {"ids": list(rows), "rows": list(rows.values())}})

    def sync(self) -> dict:
        return {t: self.push(t)["rows"] for t in sorted({f["tag"] for f in self.facts})}

    def forget(self, fid: int) -> dict:
        with self.lock:
            f = next((x for x in self.facts if x["id"] == fid), None)
            if f is None:
                return {"error": f"no fact {fid}"}
            f["state"] = "forgotten"
            rows = self.push(f["tag"])["rows"]
            self.save()
            return {"id": fid, "forgotten": True, "tag": f["tag"], "tag_rows": rows}

    # -- lessons ----------------------------------------------------------------------------------------------
    def lesson(self, fact: dict) -> dict:
        text = fact["text"]
        qs = fact.get("train") or numbered(self.e.ask(QUESTIONS.format(fact=text, n=self.a.questions + self.a.held)))
        if len(qs) < 3:
            raise RuntimeError("too few questions")
        if fact.get("train"):
            train = [(q, a) for q, a in fact["train"]]
            held = fact.get("held", [])
        else:
            random.Random(len(self.facts)).shuffle(qs)
            held, tq = qs[:self.a.held], qs[self.a.held:]
            train = [(q, self.e.ask(ANSWER.format(fact=text, q=q), max_tokens=60)) for q in tq]
        subject = fact.get("subject") or self.e.ask(SUBJECT.format(fact=text), max_tokens=24).strip(" \"'.?")
        subject = key_for(text, subject, [q for q, _ in train] + held)
        besides = numbered(self.e.ask(BESIDES.format(fact=text, n=2)))[:2]
        twins = numbered(self.e.ask(TWINS.format(qs="\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(train[:2])))))[:4]
        # twins (another thing of the same kind) and keep prompts must keep their replies; questions about the same
        # subject that the fact does not answer may change (the model now knows more about it) and are reported only
        return {"text": text, "subject": subject, "train": train, "held": held, "besides": besides, "twins": twins,
                "keep": random.Random(7).sample(KEEP, self.a.keep)}

    def prompt_ids(self, q: str, subject: str) -> tuple[list[int], list[int]]:
        text = TEMPLATE.format(q=q)
        enc = self.tok.encode(text, add_special_tokens=False)
        at = text.lower().find(subject.lower())
        pos = [i for i, (x, y) in enumerate(enc.offsets) if at >= 0 and y > at and x < at + len(subject) and y > x]
        return enc.ids, pos

    def rows_for(self, train: list[tuple[list[int], list[int], list[int]]]) -> np.ndarray:
        """Each question's subject rows that see only the subject (bigram rows inside it, trigram rows past its 3rd
        token), as experiments/04 chose them."""
        chosen = []
        for ids, pos, _ in train:
            ti = self.g.ids(self.g.initial_history(), np.array(ids))
            for t in pos:
                if len(pos) == 1:
                    chosen.append(ti[t])
                    continue
                if t - 1 in pos:
                    chosen.append(ti[t, :8])
                if t - 1 in pos and t - 2 in pos:
                    chosen.append(ti[t, 8:])
        return np.unique(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)

    # -- one fact ---------------------------------------------------------------------------------------------
    def learn_one(self, fid: int, fact: dict, tag: str, say) -> None:
        f = next(x for x in self.facts if x["id"] == fid)
        les = self.lesson(fact)
        f.update(subject=les["subject"], lesson=les)
        train = []
        for q, ans_text in les["train"]:
            ids, pos = self.prompt_ids(q, les["subject"])
            ans = self.tok.encode(ans_text + "<|im_end|>", add_special_tokens=False).ids
            train.append((ids, pos, ans))
        sel = self.rows_for(train)
        if len(sel) == 0:
            raise RuntimeError(f"the subject \"{les['subject']}\" is not in its questions")
        committed = self.tag_rows(tag)
        base = bf16_to_f32(self.table.rows(sel)).astype(np.float64)
        for k, i in enumerate(sel):
            if int(i) in committed:
                base[k] = bf16_to_f32(np.asarray(committed[int(i)], np.uint16)).astype(np.float64)
        s2 = float((base ** 2).mean())
        wgt = 1.0 + (self.counts[sel].astype(np.float64) / 39.0 if self.counts is not None else 0.0)
        # a twin that still says the key asks about the same thing in other words: it may change like the besides
        gated = [q for q in les["twins"] if les["subject"].lower() not in q.lower()] + les["keep"]
        before = {q: self.e.ask(q, tag, 80) for q in gated}
        delta, m1, m2 = np.zeros_like(base), np.zeros_like(base), np.zeros_like(base)
        cand = lambda: {"ids": sel.tolist(), "rows": bf16_rne((base + delta).astype(np.float32)).tolist()}  # noqa: E731
        lessons = [{"tokens": ids + ans[:-1], "at": list(range(len(ids) - 1, len(ids) + len(ans) - 1)), "targets": ans}
                   for ids, _, ans in train]
        steps, t0 = 0, time.time()
        state = {"m1": m1, "m2": m2, "delta": delta}

        def train_steps(n: int, stop_when_greedy: bool) -> None:
            nonlocal steps
            for _ in range(n):
                steps += 1
                r = self.e.op({"op": "grad", "tag": tag, "lessons": lessons, "sel": sel.tolist(), "overlay": cand()})
                grad = np.asarray(r["grad"], np.float64) / len(train)
                grad += self.a.lam * 2 * wgt[:, None] * state["delta"] / (state["delta"].size * s2)
                state["m1"] = 0.9 * state["m1"] + 0.1 * grad
                state["m2"] = 0.999 * state["m2"] + 0.001 * grad ** 2
                state["delta"] -= self.a.lr * (state["m1"] / (1 - 0.9 ** steps)) / (np.sqrt(state["m2"] / (1 - 0.999 ** steps)) + 1e-8)
                delta[:] = state["delta"]
                hits = [self.e.op({"op": "hits", "tag": tag, "prompt": ids, "cont": ans, "overlay": cand()}) for ids, _, ans in train]
                if steps % 5 == 0:
                    say("progress", id=fid, step=steps, loss=round(r["loss"] / len(train), 3),
                        greedy=[h["greedy_hits"] for h in hits], of=[h["of"] for h in hits])
                if stop_when_greedy and all(h["greedy_hits"] == h["of"] for h in hits):
                    return

        def verify() -> tuple[list[bool], list[bool], list[str]]:
            # through the server's chat, the candidate rows over the tag's in the scratch tag
            self.push(SCRATCH, {**committed, **{int(i): v for i, v in zip(cand()["ids"], cand()["rows"])}})
            judge = lambda q: self.e.ask(JUDGE.format(fact=les["text"], q=q, reply=self.e.ask(q, SCRATCH, 80)), max_tokens=3).lower().startswith("yes")  # noqa: E731
            rec = [judge(q) for q, _ in les["train"]]
            hel = [judge(q) for q in les["held"]]
            dmg = []
            for q, a0 in before.items():
                a1 = self.e.ask(q, SCRATCH, 80)
                if a1 != a0 and self.e.ask(SAME.format(q=q, a=a0, b=a1), max_tokens=3).lower().startswith("yes"):
                    dmg.append(q)
            self.push(SCRATCH, {})
            return rec, hel, dmg

        # rounds, as Living Weights takes them: train until every answer is greedy, check through chat; while a
        # question does not come back and nothing was damaged, train on (past greedy) and check again
        train_steps(self.a.steps, True)
        recalled, held, damage = verify()
        for _ in range(self.a.rounds):
            if damage or (all(recalled) and (not held or sum(held) >= len(held) / 2)):
                break
            say("progress", id=fid, step=steps, round="again", recalled=sum(recalled), of=len(recalled))
            train_steps(self.a.round_steps, False)
            recalled, held, damage = verify()
        ok = all(recalled) and not damage and (not held or sum(held) >= len(held) / 2)
        f.update(steps=steps, seconds=round(time.time() - t0, 1), recall=sum(recalled) / len(recalled),
                 held=(sum(held) / len(held)) if held else None, damage=damage)
        if ok:
            f.update(state="learned", rows={str(i): v for i, v in zip(cand()["ids"], cand()["rows"])})
            self.push(tag)
            say("learned", id=fid, recalled=True, steps=steps)
        else:
            why = (f"near misses or keep prompts changed: {damage[:2]}" if damage else
                   "its held-out questions did not come back" if all(recalled) else "it did not come back")
            f.update(state="missed")
            say("learned", id=fid, recalled=False, steps=steps, message=why)
        self.save()

    def learn(self, body: dict, say) -> None:
        tag = body.get("tag", "")
        if tag == SCRATCH:
            return say("failed", message=f"tag {SCRATCH} is the learner's own")
        with self.lock:
            if body.get("facts"):
                facts = [x if isinstance(x, dict) else {"text": x} for x in body["facts"]]
            else:
                text = body.get("text") or ""
                if not text.strip():
                    return say("failed", message="text is required")
                facts = [{"text": t} for t in self.e.ask(FACTS.format(text=text), max_tokens=600).splitlines() if t.strip()]
            if not facts:
                return say("failed", message="no facts found in the text: nothing was learned")
            ids = []
            for fact in facts:
                fid = 1 + max((x["id"] for x in self.facts), default=0)
                self.facts.append({"id": fid, "text": fact["text"], "tag": tag, "source": body.get("source", "chat"),
                                   "state": "queued", "at": int(time.time())})
                ids.append(fid)
                say("fact", id=fid, text=fact["text"])
            self.save()
            for fid, fact in zip(ids, facts):
                say("learning", id=fid)
                try:
                    self.learn_one(fid, fact, tag, say)
                except Exception as ex:  # noqa: BLE001 - the stream reports it and goes on
                    next(x for x in self.facts if x["id"] == fid)["state"] = "failed"
                    self.save()
                    say("learned", id=fid, recalled=False, steps=0, message=str(ex)[:300])

    def graph(self) -> dict:
        keys = ("id", "text", "tag", "source", "state", "subject", "steps", "seconds", "recall", "held", "at")
        return {"facts": [{k: f.get(k) for k in keys} | {"rows": len(f.get("rows", {}))} for f in self.facts]}


def serve(learner: Learner, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def reply(self, code: int, obj: dict) -> None:
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/slide/graph":
                return self.reply(200, learner.graph())
            self.reply(404, {"error": {"message": f"unknown path {self.path}"}})

        def do_DELETE(self):  # noqa: N802
            m = re.fullmatch(r"/v1/slide/facts/(\d+)", self.path)
            if m:
                r = learner.forget(int(m.group(1)))
                return self.reply(404 if "error" in r else 200, r)
            self.reply(404, {"error": {"message": f"unknown path {self.path}"}})

        def do_POST(self):  # noqa: N802
            if self.path == "/v1/slide/sync":
                return self.reply(200, {"tags": learner.sync()})
            if self.path != "/v1/slide/learn":
                return self.reply(404, {"error": {"message": f"unknown path {self.path}"}})
            body = self.body()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def say(kind: str, **fields) -> None:
                try:
                    self.wfile.write(f"event: {kind}\ndata: {json.dumps(fields, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                except OSError:
                    pass  # the client left; learning goes on

            learner.learn(body, say)
            try:
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                pass

        def log_message(self, fmt, *args):  # noqa: A002
            print(f"learnd: {fmt % args}", flush=True)

    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--state", type=Path, default=root / "state")
    ap.add_argument("--model", type=Path, default=paths.MODEL)
    ap.add_argument("--counts", type=Path, default=root / "data/rowfreq/counts.npy")
    ap.add_argument("--questions", type=int, default=4, help="training questions a fact")
    ap.add_argument("--held", type=int, default=2, help="held-out questions a fact")
    ap.add_argument("--keep", type=int, default=4, help="keep prompts checked a fact")
    ap.add_argument("--steps", type=int, default=40, help="most steps of the first round")
    ap.add_argument("--rounds", type=int, default=2, help="further rounds while a question does not come back")
    ap.add_argument("--round-steps", type=int, default=8, help="steps a further round")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--no-sync", action="store_true", help="do not push the saved tags to the server at start")
    a = ap.parse_args()
    learner = Learner(a)
    if not a.no_sync and learner.facts:
        print(f"learnd: synced tags {learner.sync()}", flush=True)
    print(f"learnd: {learner.e.model} at {a.engine}; listening on :{a.port}", flush=True)
    serve(learner, a.port)


if __name__ == "__main__":
    main()
