# Recipe: ngram-learn on a DGX Spark

From a Spark with nothing of ours on it to a server that learns facts. About an hour, most of it the first engine
build and the checkpoint download.

## 1. What the Spark needs

- A DGX Spark (GB10, sm_121) with Docker and the NVIDIA container runtime; your user in the `docker` group.
- ~200 GB free disk: the checkpoint (~122 GiB), MiaAI-Lab's image, our builds (~3 GB each).
- [uv](https://docs.astral.sh/uv/) on the host for the tools (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## 2. The base image and the checkpoint (MiaAI-Lab's recipe)

Our engine is TensorFold's `zig-flashnext` branch at `db28187` with MiaAI-Lab's Flash Next CUDA patches (MIT); their
recipe builds the image we build on and fetches the checkpoint:

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold.git
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold && scripts/prepare.sh
```

That leaves the image `tensorfold-qwen38:zig-db28187` and the checkpoint `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`
(revision `1464274120d3…`) in a Hugging Face cache. Point `HF_DIR` at that cache's root (the directory holding
`hub/`); ours is `~/qwen38/hf`, the default.

## 3. The fork and the harness

```bash
git clone -b ngram-learn https://github.com/axonzeta/TensorFold.git
cd TensorFold/ngram-learn
ln -s .. tensorfold                       # the scripts build ./tensorfold
docker build -t ngram-learn-dev:latest -f docker/Dockerfile.dev docker   # MiaAI-Lab's image + Zig 0.17.0
mkdir -p logs .home .cache && (cd tools && uv sync)
```

## 4. Build

```bash
./dev.sh bash -lc /work/build.sh          # engine + CUDA kernels + the TP=1 Triton kernel set -> build/opt (~25 min)
./dev.sh bash -lc "cd /tensorfold && zig build test"   # host tests
```

`BUILD=name AOT=0` builds another binary into `build/name` reusing `build/opt`'s kernel set (a few minutes).

## 5. Check the engine

```bash
./probe.sh worker --eager --context 4096 --kv-dtype bf16 --kv-gib 4 < /dev/null   # loads, prints {"ready":true}
(cd tools && uv run batch_check.py)       # a batched training pass equals its lessons one by one: BATCH-OK
```

## 5b. Row frequencies (optional, recommended)

learnd weights each row's change by how often the corpus looks it up, so common rows move least. Without
`data/rowfreq/counts.npy` the penalty is unweighted (learnd says so at start). To build it (~6 GB of text, 41 min of
CPU, a 1.28 GB file; experiments/01b-row-frequency):

```bash
data/rowfreq.sh                           # FineWeb, UltraChat and code samples into data/bg, then the counts
```

## 6. Serve with learning

```bash
LEARN=1 PORT=8890 ./serve.sh              # docker container "ngram-learn-serve"; ~2 min to load
docker logs -f ngram-learn-serve          # "ngram-learn: learn ops on POST /v1/ngram/op, overlays by metadata.tag"
(cd tools && uv run learnd.py --engine http://127.0.0.1:8890 --port 8091 --state ../state)
```

Then follow [walkthrough.md](walkthrough.md). Without `LEARN=1` the server is MiaAI-Lab's, with
`TF_NGRAM_OVERLAY=file.safetensors` applying a fixed overlay to every request.

## Settings that matter

| Setting | Default | Why |
| --- | --- | --- |
| `LEARN` (serve.sh) | 0 | 1 adds the learn ops and overlays by tag |
| `RESERVE_GIB` (serve.sh) | 26 with `LEARN=1` | memory kept from the sequences' budget for the training pass (~14 GiB) |
| `CONTEXT`, `PARALLEL`, `KV_DTYPE` | 32768, 1 (2 with `LEARN=1`), fp8 | the sequences' budget; with learning, a second stream keeps requests from waiting behind learnd's own chats |
| `--lr`, `--steps`, `--rounds` (learnd) | 2e-3, 40, 2 | ~28 steps a fact; 6e-3 is no faster here and generalises worse (0.70 vs 0.78 held-out) |
| `--questions`, `--held`, `--keep` (learnd) | 4, 2, 4 | lesson size; each training question adds rows and pass time |

## Troubleshooting

- **`{"error":"NoLearner"}` (404)**: the server was started without `LEARN=1`, or with two ranks.
- **`CudaFailed` from a learn op, then from chat**: an intermittent GPU fault in the training pass (Xid 31, an
  illegal read) kills the server's CUDA context; restart the server and `POST /v1/slide/sync` to learnd (it also
  syncs at its own start). This is the open RC1 issue: see PLAN.md's Status.
- **A fact is `missed`**: `GET /v1/slide/graph` shows why; usually the subject is not in the model-written questions
  (pass `subject` with the fact) or near misses changed (the fact is too generic for its subject).
