# ngram-learn

Continual learning of facts into Qwen3.8 Flash Next's n-gram memory (its Engram-style hashed embedding table), served
by TensorFold's Zig engine on a DGX Spark (GB10, CUDA).

A fact is learned as a small set of replacement table rows (an overlay), found by training through the model with a
backward pass that runs inside the serving engine on the served weights, as TensorFold's Living Weights does for
Nemotron on Metal. No second copy of the model and no dense weight edits.

| Path | What it holds |
| --- | --- |
| [RELEASE-rc3.md](RELEASE-rc3.md) | The current release candidate: what is in it, results, limits |
| [RELEASE-rc2.md](RELEASE-rc2.md) | RC2 |
| [PLAN.md](PLAN.md) | The work, in phases, with exit criteria |
| [docs/flashnext-ngram-memory.md](docs/flashnext-ngram-memory.md) | The n-gram memory's exact forward and gradient, with source references |
| [docs/living-weights-map.md](docs/living-weights-map.md) | Living Weights' Nemotron/Metal modules and their Flash Next/CUDA counterparts |
| [docs/references.md](docs/references.md) | Papers and prior work |
| [docs/phase3-backward.md](docs/phase3-backward.md) | The in-engine training pass: building blocks, served capture, batching |
| [docs/serving.md](docs/serving.md) | Learning in the server: learn ops, overlays by tag, learnd's API |
| [docs/walkthrough.md](docs/walkthrough.md) | A session: facts learned from text, read back by tag, deleted |
| [docs/upstream.md](docs/upstream.md) | Upstream changes and what they mean here |
| `tools/` | Host-side tools: row ids, frequencies, coverage, editing drivers, checks, `learnd.py` (the learner) |
| `patches/` | The engine changes as files and apply scripts (the fork's `ngram-learn` branch holds them as commits) |
| `experiments/` | One folder per experiment: command, pins, results |
| `tensorfold/` | The engine fork, on the Sparks: github.com/axonzeta/TensorFold, branch `ngram-learn` |
