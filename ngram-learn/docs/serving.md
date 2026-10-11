# Learning in the server

The fork's server learns facts while it serves. Two processes run on the Spark:

```
 clients ──chat (metadata.tag)──▶ tensorfold-native serve  (TF_NGRAM_LEARN=1)
                                    │  engine thread: serving rounds; when no stream lives, one learn op
                                    │  overlays by tag: rows that replace n-gram table rows for tagged requests
 clients ──/v1/slide/*──▶ learnd ──POST /v1/ngram/op──┘ (grad, hits, set, load, status)
                          └──chat──▶ the server (lessons written by the model, recall and damage checks)
```

- **The server** owns the weights and the GPU. It adds learn ops on `POST /v1/ngram/op`, which wait in a queue and run
  on the engine thread only while no request is live, and n-gram overlays by tag. A request picks an overlay with
  OpenAI's `metadata` map: `"metadata": {"tag": "work"}`. No tag (or `""`) reads the default overlay.
- **learnd** (`tools/learnd.py`) is the learner: it writes a lesson with the served model, trains the fact's rows with
  `grad` and `hits` ops, checks the result through the server's own chat and commits the rows into the fact's tag.
  It keeps the fact registry and the rows on disk (`state/facts.json`) and pushes them again after a server restart.

The two-process split keeps the engine change small (no lesson logic or optimiser in Zig) and keeps the learner's API
easy to move: Living Weights is being reworked upstream (see [upstream.md](upstream.md)), and learnd mirrors its
current `/v1/slide/learn` behind an adapter.

## The server's learn ops

`POST /v1/ngram/op` with one JSON object; the reply is one JSON object (`{"error": ...}` with status 400 on failure,
404 when the server was started without `TF_NGRAM_LEARN=1`).

| Op | Body | Reply |
| --- | --- | --- |
| `grad` | `tag`, `lessons: [{tokens, at, targets}]`, `sel: [row ids]`, `overlay?: {ids, rows}` | `loss` (summed NLL), `grad` (each `sel` row's gradient) |
| `hits` | `tag`, `prompt`, `cont`, `overlay?` | `greedy_hits`, `first_miss`, `of`: teacher-forced greedy check |
| `set` | `tag`, `clear?`, `overlay?: {ids, rows}`, `copy?: [[dst, src]]`, `drop?: [ids]` | `tag`, `rows` |
| `load` | `tag`, `path`, `clear?` | rows from a safetensors overlay (`ids` I64, `rows` BF16) |
| `status` | | ops run, rows per tag |

Rows are bf16 bit patterns (integers), 160 per row. `grad` and `hits` read the tag's overlay with the op's own
`overlay` over it, so a candidate can be scored without committing it.

How it runs inside the engine (`families/flashnext/ngram_service.zig`):

- `LaneHost.run` calls the learn hook only when no stream is active, filling or queued; one op per call, so a request
  that arrives waits at most for the op in flight (a grad pass: ~5 s on 3 short questions).
- An op binds the engine's own sequence, turns CUDA graphs, MTP absorption and sampling off and restores all of them
  after. The served capture for `grad` is the engine's normal prompt pass with the capture hook set.
- A tag's overlay is created once and never freed (a sequence holds a pointer to it); `set` with `clear` empties it.
  Each sequence takes its tag's overlay at admission; the host n-gram gather reads rows per sequence.
- A tagged prompt is never kept or resumed by the prompt cache, and every kept prompt state is dropped after any
  overlay change, so no reply runs on rows that no longer exist.
- Only one rank (TP=1): the overlay lives in the host gather.

Memory: the training pass holds about 14 GiB beside the serving engine (the layer scratch, routed experts dequantised
for one layer, the bf16 head, transposed FP8 faces). `serve.sh` with `LEARN=1` keeps 26 GiB out of the sequences'
budget (`TENSORFOLD_MEMORY_RESERVE_GIB`); with `--context 32768`, `--parallel 1` and an fp8 KV cache that leaves
~19.7 GiB of sequence memory.

## learnd's API

| Route | Body | Reply |
| --- | --- | --- |
| `POST /v1/slide/learn` | `{"text", "source"?, "tag"?}` or `{"facts": [{"text", "subject"?, "train"?, "held"?}], "tag"?}` | SSE: `fact {id, text}`, `learning {id}`, `progress {id, step, loss, greedy, of}`, `learned {id, recalled, steps, message?}`, `failed {message}`, then `[DONE]` |
| `GET /v1/slide/graph` | | every fact: id, text, tag, state, subject, steps, seconds, recall, held, rows |
| `DELETE /v1/slide/facts/<id>` | | the fact's rows leave its tag; the tag's other facts are applied again |
| `POST /v1/slide/sync` | | every tag pushed to the server again |

`learn` events match Living Weights' (`fact`, `learning`, `learned`, `failed`); `progress` is ours.

### What a fact goes through

1. **Facts.** With `text`, the model lists the facts worth keeping (one short sentence each). With `facts`, they are
   taken as given; `train` (question, answer pairs), `held` and `subject` may be given too.
2. **Lesson** (Living Weights' prompts): 4 questions the teller might ask and 2 held out, a one-sentence answer to
   each, 2 questions about the same subject that the fact does not answer, 2 twins about another thing of the same
   kind, and 4 fixed keep prompts. The fact's **key** is the longest run of the fact's own words (not stop words)
   that most questions contain: "Kestrel" for "Project Kestrel launches on 3 November" asked as "When does Kestrel go
   live?", "dog" for "My dog is called Biscuit" asked as "What's my dog called?".
   **Facts about the user** (in the first person: "My dog is called Biscuit.") take a different lesson: the model
   lists the 2–4 phrases people use to ask about it ("my dog", "my dog's name", "my puppy"), never the answer; the
   training questions must each use one; the held-out questions are written freely. Every phrase is a key, and the
   fact trains without the frequency penalty: its tag is the user's own (experiments/06-user-facts).
3. **Rows.** In each training question, the subject's rows that see only the subject: bigram rows whose previous
   token is in the subject, trigram rows whose two previous tokens are (all 16 when the subject is one token). Their
   current values: the tag's committed rows where it has them, else the table's.
4. **Training.** Adam on the rows (lr 2e-3), all training questions in one batched `grad` op a step, a penalty on
   the change weighted by each row's corpus frequency; after each step a `hits` op per question; it stops when every
   answer is greedy (at most 40 steps).
5. **Checks** through the server's chat, with the candidate in a scratch tag (`_learnd`): each training and held-out
   question's reply is judged by the model against the fact; the twins' and keep prompts' replies are compared with
   their replies before the edit, and a changed reply counts as damage when the model judges it tells the user
   something the old one did not. Questions about the same subject that the fact does not answer may change (the
   model now knows more about the subject) and are not gated, nor are twins that still contain the key.
6. **Rounds** (as Living Weights takes them): while a question does not come back through chat and nothing was
   damaged, train 8 more steps past greedy and check again, at most twice.
7. **Commit** when every training question comes back, at least half the held-out ones do and nothing was damaged:
   the rows join the tag's overlay and the registry. Otherwise the fact is kept as `missed` with the reason, and
   nothing changes in the server.

`DELETE` re-applies the tag's remaining facts in learning order, so a row two facts share takes the later fact's
value. A fact learned after another that shares rows starts from the earlier fact's rows.

## Run it

```
# on the Spark, in ngram-learn/
BUILD=serve LEARN=1 PORT=8890 ./serve.sh                 # the server (docker, ~2 min to load)
cd tools && uv run learnd.py --engine http://127.0.0.1:8890 --port 8091 --state ../state

curl -N localhost:8091/v1/slide/learn -d '{"text": "Our team decided that Project Kestrel launches on 3 November.", "tag": "work"}'
curl localhost:8890/v1/chat/completions -d '{"model": "flashnext", "messages": [{"role": "user",
  "content": "When does Project Kestrel launch?"}], "metadata": {"tag": "work"}}'
```

See [walkthrough.md](walkthrough.md) for a full session with outputs.

## Limits (RC1)

- One rank only; the training pass blocks the engine thread for one op at a time (~5 s), so a request arriving
  mid-op waits for it.
- A one-token subject ("Kestrel") trains rows that also fire in other contexts with that token: in the walkthrough,
  "the Kestrel rocket" picks up the launch date. Longer subjects keep to themselves (context-free rows).
- Facts in one tag that share rows are not re-solved jointly (EngramEdit-style targets are Phase 4's second version).
- A server restart forgets the overlays until learnd pushes them (`learnd` does at start; `POST /v1/slide/sync`).
