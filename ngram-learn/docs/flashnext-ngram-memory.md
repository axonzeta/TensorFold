# Flash Next's n-gram memory

Verified against code on 2026-10-10. Paths: **Py** = `python-0.6:src/tensorfold/families/qwen4_exp/model.py`,
**ML** = `python-0.6:.../qwen4_exp/model_layers.py`, **Z** = `zig/src/families/flashnext/` (upstream `zig-flashnext`),
**K** = `zig/kernels/metal/flashnext/`, **Mia** = MiaAI-Lab's CUDA patches (`zig/src/families/flashnext/cuda_*.zig`).

## Where it sits

- One site: `language_model.model.layers.1.ple` (config `ple_layer_ids=[2]`, one-based), the second decoder layer.
- The MTP draft layer has no n-gram memory (`python-0.6:.../qwen4_exp/mtp.py:28`); it sees its effect only through
  the target's streams.

## Row ids

- `ngram_size=3`, `heads_per_ngram=8`: 16 heads, 0-7 bigram, 8-15 trigram. Row width 2560 / 16 = 160 (Py:91,108).
- Head sizes: the h-th primes after 19,999,999 (20000003 ... 20000171); offsets cumulative; 320,001,446 rows used,
  padded to 320,001,536 (Py:94-99).
- Multipliers from splitmix64 with seed 1234: [23703573157769, 20109073645365, 8052911324071] (Py:76-80,
  Z/config.zig:341-357).
- `s_p = tok[t-p]` inside the current EOS segment, else `eos` (248044; `<|im_end|>` 248046 does not reset).
  `bi = s0*m0 ^ s1*m1`, `tri = bi ^ s2*m2` (wrapping int64), `id_h = floormod(mix, size_h) + offset_h`
  (Py:120-138; Z/replay.zig:105-127 `fz_ple_ids`; Mia `cuda_ngram.zig`).
- All 8 heads of an order share `mix` and differ only in modulus. Token ids are hashed raw: no case or Unicode
  normalisation (DeepSeek's Engram normalises; Flash Next does not appear to).
- History is the last 2 tokens, carried across requests in the cache; Zig snapshots keep it plus a 9-row conv tail
  (Z/snapshot.zig:13,155,212).

## Forward (Py:184-202, ML:126-199)

S = 4 hyper-connection streams, D = 2560, h in R^(S*D). CRMS = per-2560-group RMSNorm * (1 + w), fp32.

```
e   = concat_h(scale * row_h)              in R^2560         (concat of 16 rows of 160)
k   = CRMS_s(W_k e)      W_k: [10240, 2560]                  (key per stream)
v   = W_v e              W_v: [2560, 2560]                   (one value for all streams)
q   = CRMS_s(h)                                              (query per stream, from the residual)
x_s = <k_s, q_s> / sqrt(2560)
g_s = sigmoid(sign(x_s) * sqrt(max(|x_s|, 1e-6)))            (one scalar per stream per position)
G_s = g_s * v
c   = CRMS_s(G)
y_t = sum_{j=0..3} w_j (.) c_{t - 3(3-j)}                    (depthwise causal conv, dilation 3, tail 9)
out = G + silu(y)
h  <- h + out                                                (all 4 streams, no inject scaling)
```

The Zig engine fuses W_k and W_v into one 12800-row projection `ple.kv` (Z/pack.zig:486-488).

## Reach of one row

A token at position t enters the bigram ids at t and t+1 and the trigram ids at t, t+1, t+2. A row looked up at t
reaches the residual at t directly and at t+3, t+6, t+9 through the conv branch.

## Gradient to a row

Given dL/d out_t (10240 wide, per position):

```
dL/dG_t   = dL/dout_t + sum_j J_crms(G_t)^T [ w_j (.) silu'(y_t') (.) dL/dout_t' ],   t' = t, t+3, t+6, t+9
dL/dg_s   = <dL/dG_ts, v>
dL/dx_s   = dL/dg_s * g_s(1 - g_s) / (2 sqrt|x_s|)       (0 when |x_s| < 1e-6)
dL/dv     = sum_s g_s dL/dG_ts
dL/dk_s   = dL/dx_s * q_s / sqrt(D)
dL/de     = W_v^T dL/dv + W_k^T J_crms(W_k e)^T dL/dk
dL/drow_h = scale * (dL/de)[160h : 160h + 160]
```

The gate depends on the rows (through the direction of W_k e), so all 16 rows at a position interact through it; the
value path is linear in the rows.

## Storage

- 128 shards of 2,500,012 rows: `shard = id // 2500012`, `local = id % 2500012`. Tensor names
  `language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.{weight,scales,biases}` or `.shards.{i}`
  (Z/index.zig:60-72).
- Formats: MLX affine 6-bit g32 (fixture), MLX 4-bit; ModelOpt BF16 / FP8 e4m3 with a per-table scale / NVFP4; EXL3
  per-row trellis; GGUF IQ4_NL (26.8 GiB). The overlay keeps edited rows in bf16 and never re-quantises the table.
- Metal Zig loads all shards into 8 GPU buffers of 16 shards (Z/replay.zig:1021-1037).
- CUDA (fork, TP=1): row ids are computed on the host per window (`cuda_forward.zig:1185-1201`, `NGram.ids` in
  `cuda_ngram.zig`); the INT4-AutoRound checkpoint's table is FP8 in `ple-table/`, memory-mapped on the host, and
  `NgramTable.gather` (`cuda_weights.zig:152`) decodes each row through a 256-entry LUT (scale folded in) to bf16, then
  the window's rows are uploaded. The overlay at TP=1 is a host-side substitution in `gather`. With two ranks the
  table lives on the GPU and `fn_ngram_gather` (`fn_pack.cu`) gathers it.
- Usage (recipe probe): a 2048-token chunk reads 32,768 rows, ~25,000 distinct.
