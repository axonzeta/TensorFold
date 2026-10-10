# Upstream watch

`tools/upstream_watch.sh` lists ashhart/TensorFold commits (main, zig-flashnext) and PRs since the last check, flagging
PRs that touch Flash Next, CUDA, n-grams, learning or drafters, plus MiaAI-Lab's Flash Next recipe. Run it at each
milestone; record here what changes our plan.

## 2026-10-10 (upstream 1.0.5)

Our base: `db28187` + MiaAI-Lab patches 0001–0009 (Flash Next CUDA, out of tree). Upstream main is ~150 commits on.

| Upstream | What it means for ngram-learn |
| --- | --- |
| #591 (open, ZackSample) Living Weights learns on CUDA (GB10): CUDA training kernels for Nemotron checked against float64; the learner made generic over backends (`learner.Of(train)`) | The integration point for our work: an n-gram overlay is another kind of learned change behind the same lessons, checks, undo and `/v1/slide/learn`. Kernels overlap with ours (`train.cu` vs `fn_train.cu`): propose sharing. |
| `2f8284e` DeviceBuffer.upload returns once its bytes have landed; `d7469ca` fp8-lane launches wait for uploads | The same upload-ordering race we hit; our `Ops.upload` stream sync is a workaround a rebase makes redundant. |
| #587 (merged) one quantized-weight interface `cuda/qlinear.zig`; #595, #602, #605 (open) NVFP4 / affine-4 / grouped experts behind it | The seam our `train_weights.zig` and MoE training path should sit on upstream. |
| #572 (merged) logprobs on the native CUDA server | Overlaps our `score` probe; use it where it fits. |
| #476 (open) Zig CUDA Flash Next, drafts off (a different implementation from MiaAI-Lab's) | Upstream still has no Flash Next CUDA; our engine dependency stays out of tree for now. |
| #599 (open) Living Weights keeps a lesson only when every fact came back; `a060f31` leak check ignores everyday words | Lesson policy we should match in Phase 4. |

Plan consequences: RC1 stays on the current base; Phase 4 mirrors Living Weights' API and learner interface so the
move to `learner.Of` is mechanical; an RFC issue upstream (draft, awaiting go-ahead) proposes the n-gram overlay as a
learned-change kind and shared CUDA training kernels.

### 2026-10-10 evening: 1.0.5 released (`bec00ae`)

ashxhart's 1.0.5 post: GLM-5.3-Flash on one 256 GB M5 Ultra, Nemotron chat reuse (19.8k-token turn prefills in
0.16 s), fixes; and "I'm re working the Living Weights which will be…" (the rest is in the post's long form, not in
the public embed). Living Weights is being reworked, so its API may move: our sidecar mirrors today's
`/v1/slide/learn` (`text`, `source`; SSE `fact`, `learning`, `learned {recalled, steps, message}`, `failed`) behind a
thin adapter, and the engine side stays our own (`POST /v1/ngram/op`), so a changed upstream API costs an adapter,
not an engine change. The "learned" commits in 1.0.5 (`7ae417a`…`91b6678`) are the prompt cache's on-disk states
(`--learn`), not Living Weights.
