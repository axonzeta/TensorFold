# 01e — Causal tracing: does Flash Next read facts from its n-gram memory at the subject?

Engine: fork + `patches/apply_probe.py` (Forward.ple_mute: zero the gated n-gram rows at chosen positions, which
removes those positions' lookups from the direct add and from the dilated conv taps) + `tensorfold score`
(`patches/flashnext_probe.zig`: prefill, teacher-forced continuation, host log-softmax in fp32). bf16 KV, eager.

Cases: `tools/score_cases.py build`, CounterFact cases 0–399 as plain completions (no chat template), true answer as
the continuation; 176 facts kept (those with enough non-subject positions). Five conditions per fact: none, subject
(every subject token), random (same count of non-subject prompt positions), tail (same count at the prompt's end),
all. 880 cases in one load (88 s) — `./probe.sh score ...`.

Δ log p(true answer) against no mute (`summary.json`):

| Facts | n | subject | random | tail | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 176 | mean −1.38 (median −0.72) | −0.21 (−0.08) | −0.24 (−0.10) | −1.52 (−1.04) |
| known (greedy = true answer) | 31 | mean −2.02 (median −0.43) | −0.20 (−0.03) | −0.13 (−0.05) | −1.96 (−0.42) |

Share of known facts still answered greedily: subject muted 58%, random 77%, tail 84%, all 58%.

**Go.** Muting the n-gram memory at the subject's positions costs about 6–10× more answer log-probability than muting
as many other positions, and accounts for about 90% (all facts) to 100% (known facts) of the effect of muting the
memory everywhere. Flash Next reads factual associations from its layer-1 n-gram memory at subject positions, as
EngramEdit and DeepSeek's Engram ablation predict.

Caveats: plain-completion prompts on a chat model (only 31 of 176 facts reproduced greedily); one checkpoint (INT4
experts, FP8 table); subject tokens are content words while random/tail positions include function words.
