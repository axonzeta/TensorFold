# Walkthrough: teaching the served model, by tag

One session on a DGX Spark with the RC1 build: the server started with `LEARN=1`, learnd beside it, two texts
taught, the answers read back with and without their tags, one fact deleted. Everything below is what the session
printed (`tools/walkthrough.py`; the full transcript is `experiments/04-chat-facts/walkthrough-transcript.md`).
Replies are cut at 160 characters.

## 0. Start

```bash
LEARN=1 PORT=8890 ./serve.sh                       # ~2 min to load
(cd tools && uv run learnd.py --engine http://127.0.0.1:8890 --port 8091 --state ../state)
```

## 1. Before

The model knows nothing about our project and refuses or guesses:

| Question | no tag | tag `work` |
| --- | --- | --- |
| When does Project Kestrel launch? | There is no widely known public project, product, or initiative officially named **"Project Kestrel"** that has a publicly announced launch date. | (the same) |
| Who leads the Kestrel project? | The Kestrel project is led by **Dr. Sam Altman**, who serves as the CEO of OpenAI. | (the same) |

## 2. Teach a text

```bash
curl -N localhost:8091/v1/slide/learn -d '{"text": "Notes from Monday'\''s planning call: Project Kestrel now
  launches on 3 November, and Mira Okafor will lead the Kestrel project from next week.", "tag": "work"}'
```

```
fact: {"id": 1, "text": "Project Kestrel launches on 3 November."}
fact: {"id": 2, "text": "Mira Okafor will lead the Kestrel project starting next week."}
learning: {"id": 1}
learned: {"id": 1, "recalled": true, "steps": 37}
learning: {"id": 2}
learned: {"id": 2, "recalled": false, "steps": 48, "message": "it did not come back"}
```

The model split the notes into two facts and learned each in turn (about 6 minutes each, with the lesson writing
and the checks):

- **Fact 1 was learned**: 24 table rows at the key "Kestrel", after 37 steps; its four training questions and both held-out questions came back through chat, and the twins and keep
  prompts answered as before.
- **Fact 2 was not**: its questions find it by "Kestrel" (the model wrote "When does Mira start on Kestrel?",
  "Who's running Kestrel now?"; "Mira" alone is in only some), so it competes with fact 1 for the same rows. Three of four questions came back, one
  of two held out; learnd kept nothing. This is the case learner v2 (targets solved jointly) is for.

## 3. A personal fact in its own tag

```
POST /v1/slide/learn {"text": "My dog is called Biscuit.", "tag": "home"}
  fact: {"id": 3, "text": "The dog's name is Biscuit."}
  learned: {"id": 3, "recalled": false, "steps": 56, "message": "its held-out questions did not come back"}
```

Every training question came back, but none of the held-out ones: the key is "dog", a common word whose rows every
sentence about any dog shares, so the change cannot be both large enough to generalise and small enough to keep.
Facts about the user keyed on common words are RC1's known limit (named things work).

## 4. After

| Question | no tag | tag `work` | tag `home` |
| --- | --- | --- | --- |
| When does Project Kestrel launch? | There is no widely known public project … | **Project Kestrel launched on November 3, 2010.** | There is no widely known public project … |
| Who leads the Kestrel project? | … led by **Dr. Sam Altman** … | … led by **Kanye West** … | … led by **Dr. Sam Altman** … |
| When does Project Falcon launch? | There is no widely recognized public project … | (unchanged) | (unchanged) |
| What is my dog called? | I don't know what your dog is called … | (unchanged) | (unchanged) |

- The `work` tag now answers the launch date; untagged requests and the `home` tag are untouched.
- The model adds a year of its own ("2010"): the fact gave none and the rows carry "3 November" into the model's
  usual phrasing of a launch.
- "Who leads the Kestrel project?" changed under `work` too (a different made-up name): a question about the same
  subject that the fact does not answer is allowed to change, and does. learnd checks twins about *other* subjects
  and keep prompts, which did not change.

The graph learnd keeps (`GET /v1/slide/graph`):

| id | fact | tag | state | key | steps | recall | held-out | rows |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Project Kestrel launches on 3 November. | work | learned | Kestrel | 37 | 1.0 | 1.0 | 24 |
| 2 | Mira Okafor will lead the Kestrel project starting next week. | work | missed | Kestrel | 48 | 0.75 | 0.5 | 0 |
| 3 | The dog's name is Biscuit. | home | missed | dog | 56 | 1.0 | 0.0 | 0 |

## 5. Delete a fact

```
DELETE /v1/slide/facts/1 -> {"id": 1, "forgotten": true, "tag": "work", "tag_rows": 0}
```

| Question | no tag | tag `work` |
| --- | --- | --- |
| When does Project Kestrel launch? | There is no widely known public project … | There is no widely known public project … |
| Who leads the Kestrel project? | … led by **Dr. Sam Altman** … | … led by **Dr. Sam Altman** … |

The rows leave the tag and the model answers exactly as before it learned anything.

## Try it yourself

- Teach facts about named things (projects, people, products, places): they work best. Give `subject` with a fact
  when its questions will name it differently from the text.
- Keep a tag a topic (`work`, `home`, a customer, a codebase): a request reads one tag's overlay.
- `tools/rc_gate.py` checks the server: tags change only tagged replies, drafted replies equal undrafted ones.
