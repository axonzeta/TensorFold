# 00 — Baseline on spark4

Engine: fork `~/ngram-learn/tensorfold` branch `ngram-learn` = TensorFold `db28187` + MiaAI-Lab patches 0001–0009
(patch files sha256 prefix `c9af7bbae586`; MiaAI-Lab image label `4144696f21e3`). Built in `ngram-learn-dev`
(`./dev.sh /work/build.sh`): `tensorfold-native` + 320 AOT kernels (101 cubins equal to the capture's).
Checkpoint: `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` @ `1464274`. Served with `./serve.sh`: fp8 KV,
context 32768, parallel 1, greedy, thinking off, text only. Loaded in 112 s; 36.4 GiB sequence memory.

`tools/baseline.py` (run1.json), 512 max tokens, wall time per request including prefill and HTTP:

| Prompt | drafted tok/s | plain tok/s | drafted == plain |
| --- | ---: | ---: | --- |
| Python hash map | 59.0 | 32.6 | yes |
| Gated delta rule vs attention | 38.6 | 32.4 | yes |
| Lighthouse story | 37.4 | 32.7 | yes |
| Planets | 50.7 | 32.7 | yes |
| Rename user_id | 51.8 | 31.1 | yes |
| Capital of Australia | 45.3 | 32.6 | yes |

Exactness 6/6. MiaAI-Lab reports 64.4 tok/s single stream (sparkDash, hash-map prompt, engine `20e709a`); ours is
59.0 measured differently (wall time incl. prefill). To reconcile with their benchmark command later.
