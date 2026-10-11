# Plan

## Goal

A Flash Next server on one DGX Spark that learns facts while it serves. You tell it something; a few minutes later it
answers from its own n-gram memory. Each fact is a small overlay of bf16 table rows, removable one fact at a time,
selectable per request by tag, with no second copy of the model and no change to the base checkpoint.

Training runs inside the serving engine, on its own weights, while no reply is being generated: the Living Weights
pattern ([docs/living-weights-map.md](docs/living-weights-map.md)), ported from Nemotron/Metal to Flash Next/CUDA.

## Principles

- **Base bits never move.** With no overlay loaded, every reply is bit-identical to the base engine. With an overlay,
  drafted output still equals `--no-drafts` output (the overlay sits in the shared row gather, so both paths see it).
- **One copy of the weights.** The training forward reuses the serving kernels; the backward reads the same buffers.
- **Activation gradients only.** Every layer is frozen; the only parameter gradient is at the n-gram rows.
- **Train against what is served.** No proxy model, no other quantisation: the rows are fitted on the served weights.
- **Measure before building.** Phase 1 can stop the project for the price of a tokenizer and a few forward passes.
- TensorFold house rules for engine code: bf16 activations, fp32 accumulation (gradients in fp32), files of at most
  about 600 lines, one-line comments, `zig build test` passes, op-by-op checks against a reference.

## Decisions taken

| Decision | Choice | Why |
| --- | --- | --- |
| Hardware | One DGX Spark (GB10, 128 GB unified) | Deployment target; a second Spark helps for parallel dev, not required |
| Engine base | TensorFold `zig-flashnext` + MiaAI-Lab's Flash Next CUDA patches (MIT) | Only Zig CUDA Flash Next; measured at ~64 tok/s single stream |
| Checkpoint | `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` @ `1464274` | What MiaAI-Lab serves and measured (~66 GiB loaded); one weight format to support in the backward |
| Store | Overlay of bf16 rows keyed by global row id | Sparse, removable, per tag, no re-quantisation |
| Where facts are written | Rows active at subject positions | EngramEdit, ROME, Geva 2023: recall is carried from the subject |
| Trainer | In-engine backward (Phase 3); forward-only optimisation as a stopgap (Phase 2b) | One copy of weights |

## Status

- 2026-10-10: Phase 0 done except the benchmark reconciliation (experiments/00-baseline); Phase 1a row-id tool
  done and golden-tested; Phase 1c first CounterFact run done (experiments/01c-key-coverage).
- 2026-10-10: Phase 1d/1e done (`tensorfold score` + ple_mute). Causal tracing says **go**: muting subject positions
  costs 6–10x more answer log-prob than other positions, ~90-100% of muting everywhere (experiments/01e-causal-tracing).
  Phase 1b row frequencies done (experiments/01b-row-frequency).
- 2026-10-10: Phase 2 overlay done and exact (experiments/02-overlay); memory transplant flips 18% of facts with no
  training. Phase 2b forward-only editing: efficacy 0.55, generalisation 0.30, specificity unchanged (02b-zo-edit).
  Next: Phase 3 in two steps — 3R, a PyTorch reference trainer on the HF `qwen4_exp` modules (transformers 5.17 in the
  dev image) with weights dequantised on the fly, for gradient-based editing now and as the oracle for 3c; then the
  in-engine CUDA backward.
- 2026-10-10: 3R done: gradient editing efficacy 1.00, generalisation 0.74, specificity unchanged on 50 CounterFact
  edits, ~30 s a fact; overlays verified in the engine (experiments/03r-grad-edit). Nodes spark2/3/8 added
  (`setup-node.sh`). Phase 3 B1 done on spark8: every frozen weight readable as bf16 from the engine's own buffers,
  bit-exact vs the reference (FP8 faces by identity matmul, int4 experts/head by fn_train; docs/phase3-backward.md).
- 2026-10-10: Phase 3 B2–B11 done on spark8: every training kernel checked vs fp64 PyTorch; the whole in-engine
  training pass runs (2.8 s a lesson pass); the in-engine learner edits CounterFact facts as well as the reference
  (cases 0–9: 1.00 / 0.60 / 0.86 vs 1.00 / 0.70 / 0.84) with one copy of the weights (experiments/03-backward).
  Open: forcing the serving prefill's MoE routes in training; speed; Phase 4 lessons for real facts in chat form.
- 2026-10-10: served capture (training on the serving prefill's own activations and routes) and batched lessons
  (several questions in one pass, `tools/batch_check.py`): chat facts 0.94 recall, 0.75 held-out, neighbours
  unchanged, 171 s a fact (experiments/04-chat-facts). Phase 4 engine side done and pushed (`0fa070c`): learn ops
  in the server (`POST /v1/ngram/op`, run between streams), overlays by `metadata.tag`; drafted = plain under a tag;
  `tools/learnd.py` learns facts from text through the live server and deletes them (docs/serving.md).
- 2026-10-10: the intermittent GPU fault (Xid 31) found and fixed (`a44400c`): training uploads raced the
  non-blocking training stream (a pageable `cuMemcpyHtoD` returns before its DMA lands); upstream's `2f8284e`
  backported. Soak: ~1,500 passes on three nodes, no fault. RC gates: `zig build test` passes; `tools/rc_gate.py`
  (overlays by tag change only tagged replies; drafted = plain, tagged and untagged) passes on the live server.
- 2026-10-10: RC1 published (`ngram-learn-rc1`, recipe checked from a clean clone). After RC1: learner v2
  (stored targets, joint solve) prototyped and dropped: v1 keeps earlier facts better, including facts on one subject
  (experiments/05-v2). Facts about the user fixed in learnd: no penalty inside a personal tag, several keys a fact,
  model-written key phrases; "my dog", "I live", "my favourite colour" all learned through the server
  (experiments/06-user-facts). Next: more keys a fact, the scale evaluation, speed.

## Phases

Sizes are relative (S, M, L, XL); each phase ends at its exit criteria, not a date.

### Phase 0 — Base engine on the Spark (M)

1. Create `tensorfold/` here: a fork of `ashhart/TensorFold` on branch `ngram-learn`, starting at `db28187` (MiaAI-Lab's
   pin) with their nine patches applied. Record the patch set's hash (`d84d7cc22655` at the time of writing).
2. Bring in Living Weights' model-independent parts from upstream `f1aef9c`: `core/engine_api.zig` learn types,
   `core/lane_host.zig` learn scheduling, `server/slide*.zig`. Decide then whether to rebase the whole patch set onto
   current `zig-flashnext` instead; prefer the smaller diff.
3. Build with Zig 0.17.0 in NVIDIA's PyTorch container (MiaAI-Lab's `scripts/prepare.sh`), serve the INT4-AR
   checkpoint, and reproduce their single-stream decode and prefill numbers.
4. Exactness baseline: drafted equals `"draft": false` on a fixed prompt set; keep the set and output hashes in
   `experiments/00-baseline/`.
5. Locate the n-gram row gather and its dequantisation on CUDA (the table's on-disk format in this checkpoint: FP8 per
   MiaAI-Lab's README) and note it in `docs/flashnext-ngram-memory.md`.

Exit: reproducible build script, baseline numbers within 5% of MiaAI-Lab's, exactness set passing, gather located.

### Phase 1 — Evidence before training (M)

**1a. Row-id tool (S).** `tools/ngram_ids.py`: tokenizer + the hash constants → the 16 row ids per position, checked
against `cuda_ngram.zig` / `fz_ple_ids` on fixed token sequences (bit-exact).

**1b. Background row frequencies (S).** Hash 1–5 B tokens of mixed web, chat and code text into per-row counts
(320 M rows × u16 saturating = 640 MB). Gives each row's background frequency for rarity ranking (TF-IDF as in sparse
memory finetuning) and a sentence index for locality tests later.

**1c. Key coverage (S).** For CounterFact / zsRE subsets and a hand-written user-fact suite (people, projects,
identifiers, preferences), with model-written paraphrases: the share of paraphrases whose subject positions share at
least one bigram or trigram row with the fact statement, by fact type, and those rows' background frequency.
Tokenizer only; runs on a Mac.

**1d. Scoring endpoint (S).** A debug route on the engine: prompt-only forward returning the log-probabilities of given
continuation tokens. Every later evaluation uses it.

**1e. Causal tracing (M).** A debug hook that scales the n-gram block's output at chosen positions. For facts the model
already knows, measure the answer log-probability drop when muting subject positions vs other positions.

Exit (go / no-go):
- Muting the n-gram output at subject positions measurably drops known-fact answer probability, more than muting
  elsewhere. If it does not, Flash Next does not route facts through this memory the way Engram does: stop and rethink.
- For entity-type facts, a clear majority of paraphrases share a rare subject row with the statement. Record which
  fact types fail (expected: those keyed on common words such as "I like").

### Phase 2 — Overlay in the serving path (M)

1. `ngram_overlay.zig`: global row id → bf16[160]. At TP=1 the table is gathered on the host
   (`NgramTable.gather`), so the overlay is a host-side substitution there; every path (prompt, decode, verify) goes
   through it. A GPU hash table is only needed for the two-rank GPU gather.
2. Overlay file: safetensors with `ids` int64 [N], `rows` bf16 [N,160], metadata (base checkpoint revision, hash
   constants, overlay version, per-fact ownership).
3. Load at start (`--ngram-overlay DIR`), hot-swap between rounds, and per-request selection by tag (request body
   `tensorfold.tag`, header `X-TensorFold-Tag`, or model-name suffix), defaulting to none.
4. Prompt-cache safety: a kept prefix is reusable under an overlay only if none of its row ids is in the overlay;
   check from the prefix's tokens at reuse time.
5. Tests: with no overlay, bits identical to Phase 0; with an overlay, drafted equals plain; a hand-made overlay (one
   n-gram's rows copied from another's) visibly changes behaviour and removing it restores the base bits.

Exit: the tests pass; decode cost of an empty overlay and of a 100k-row overlay measured (target under 1%).

### Phase 2b — Forward-only learning (S–M, optional, in parallel with Phase 3)

Zeroth-order optimisation (MeZO-style) of a lesson's selected rows, in a 64–256-dimensional subspace, using only the
scoring endpoint: no backward. It exercises the lesson, row-selection, verify and commit logic of Phase 4 end to end
on the served weights while Phase 3 is built, and gives a first answer to "does writing these rows teach the fact".

Exit: on 20 entity facts, report recall on held-out paraphrases, locality, and minutes per fact. A weak result here
does not stop Phase 3; a strong one may change its priority.

### Phase 3 — Backward inside the engine (XL)

The core build. A training step = forward over up to 256 rows (question + answer) through every layer with the serving
kernels, cross-entropy on the answer tokens, then activation gradients back to layer 1 and into the n-gram rows.

**3a. Training forward and activation storage (M).**
- Run the prompt path for one sequence of ≤256 rows, saving each layer's input (4 × 2560 bf16 per row) and recomputing
  inside a layer during its backward, as Living Weights does with its per-layer slots (`train.zig` `copy`).
- At ≤2048 tokens the QSA indexer selects every block, so attention is plain causal GQA: no indexer backward.
- Routes are frozen: save each MoE layer's expert ids and weights.

**3b. Loss and head (S).** Chunked softmax cross-entropy over the full vocabulary on answer rows (logits never stored
whole), final norm backward, and dh = dlogits · W_head (dequantised).

**3c. Backward kernels (L).** fp32 gradients, `zig/kernels/cuda/fn_train_*.cu`, one launcher module per layer kind:

| Kernel group | What | Notes |
| --- | --- | --- |
| Transposed quantised matmul | dx = dy · W for each weight format in the checkpoint (int4 AutoRound / GPTQ g128, plus whatever the attention, GDN, shared-expert and head projections use) | The largest single item; dequantise on the fly, fixed tiles |
| RMSNorm / group RMSNorm | (1+w) norms, per-stream norms | |
| Hyper-connections | grouped norm, `silu(down/4)`, `sigmoid(up)`, stream mean, injection `2·sigmoid(inject/4)` | Equations in the recipe's `qwen4_exp_liger_hc.py` |
| QSA attention | causal GQA (24 q / 2 kv heads, head dim 256, partial RoPE over 64), sigmoid output gate in q_proj | Port Living Weights' `tf_train_attn_q` / `tf_train_attn_kv` |
| GatedDeltaNet | gated delta rule backward over ≤256 rows (recurrent with state checkpoints every 16 rows, as Living Weights' SSM backward), causal conv1d backward, gate (A_log, dt_bias) and q/k L2-norm backward | Alternative: AOT-compile FLA's Triton GDN backward through the existing `cuda_triton.zig` path |
| MoE | routed experts on frozen routes (pairs in / out), SwiGLU backward, shared expert, routing-weight backward | Port Living Weights' `moe` |
| n-gram block | the formulas in `docs/flashnext-ngram-memory.md`, gate and dilated conv included, to dL/d row | The only parameter gradient |

**3d. Checks (M).** Op by op against fp32 PyTorch on captured inputs (TensorFold's usual discipline): each backward
kernel's output within tolerance; a whole-model gradient at the n-gram rows with cosine ≥ 0.999 against a reference
on short sequences; finite-difference spot checks on the n-gram block.

**3e. Optimiser and sites (S).** `ngram_sites.zig`: the lesson's trainable rows (base rows copied to fp32), Adam (or SGD,
as sparse memory finetuning used), keep / restore for undo.

Exit: gradient checks pass; one step on 4 × 128 rows measured on the Spark; serving bits unchanged when no learning
has happened.

### Phase 4 — The learner (L)

`ngram_learner.zig` behind the lane host's learn hook; `/v1/slide/learn`-compatible API so omp's `/learn` works.

1. **Lessons.** Reuse Living Weights' model-written lessons: facts from text, K ≥ 4 paraphrases varied in wording,
   case and the word before the subject, two held-out questions, near misses, keep prompts. Add subject spans
   (Living Weights already asks the model for a question's subject).
2. **Row selection.** Rows at the subject's tokens in every paraphrase, minus chat-template n-grams, ranked by rarity
   (Phase 1b counts) and gradient magnitude; bigram rows for generalisation, trigram rows for locality; a cap per fact.
3. **Training.** First version: train the selected rows end to end on the paraphrase answers. Second version
   (EngramEdit-style): fit one target change at the n-gram block's output per subject position via the backward,
   store it, then solve the rows locally through the block (gate-aware) with a frequency-weighted penalty, jointly with
   every earlier fact that shares a row, always from the base rows. Because targets are stored, a re-solve needs no
   backward, the result does not depend on edit order, and a fact can be removed exactly.
4. **Verify.** Held-out paraphrases recall the fact; near misses and keep prompts keep their answers; background
   sentences containing the edited n-grams (from the Phase 1b index) keep their next-token distributions within a KL
   bound. Fail → undo and report why.
5. **Commit.** Write the overlay and a fact registry (fact → rows, targets, lesson); per-fact delete; compaction.

Exit, on a CounterFact-style subset and the user-fact suite:
- Efficacy ≥ 95%, held-out paraphrase recall ≥ 85%, locality within 2 points of base on neighbourhood prompts.
- 1,000 sequential edits keep ≥ 95% of base on a fixed general eval (MMLU and code subsets).
- Minutes per fact on the Spark reported; serving unaffected while idle learning runs.

### Phase 5 — Scale and operate (M)

- 5k–10k edits; compaction; overlay memory and lookup cost at scale.
- Per-tag overlays in production; metrics (facts, rows, overlay hits per request) in `/metrics` and the dashboard.
- Facts the n-gram table cannot hold (common-word subjects): route to a low-rank change (Living Weights style) or to
  prompt memory; decide from Phase 1c and Phase 4 data.
- Related track, separate plan: the drafter (MTP) does not read the n-gram memory and will lag learned facts; a
  drafter LoRA that follows the learned model, and a persistent n-gram drafter.

## Dependencies

```
Phase 0 ──► Phase 1 (go / no-go) ──► Phase 2 ──► Phase 2b ─┐
                                         │                  ├──► Phase 4 ──► Phase 5
                                         └──► Phase 3 ──────┘
```

Phase 1a–1c need no Spark and can start immediately; Phase 3c kernels can start once Phase 0 locates the formats.

## Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Flash Next does not route facts through layer-1 memory at subject positions | The approach fails | Phase 1e go / no-go before any training code |
| Raw token ids (no normalisation) and bigram-minimum keys | Paraphrases miss the fact's rows | Paraphrases varied in case, spacing and preceding word; measured in 1c |
| Common-word subjects ("I like", "my favourite") | Edits bleed everywhere | Rarity filter; route those facts elsewhere (Phase 5) |
| Row-dependent gate | Written values gated shut | Gradient includes the gate; local solve is gate-aware |
| Backward kernel effort, especially transposed quantised matmul and GDN | Phase 3 slips | Phase 2b keeps the learner moving; FLA Triton AOT fallback for GDN |
| Upstream churn (TensorFold merges dozens of PRs a week; patch set out of tree) | Rebase cost | Keep engine changes in new files; small hooks; talk to upstream early |
| Multi-hop use of learned facts | Weak (15–37% in EngramEdit) | Out of scope; report it |

## Upstream

- Share the plan with Ash Hart once Phase 1 has data: Living Weights already trains in-engine; a sparse, removable
  overlay in Flash Next's own memory is a natural next step, and the drafter lag is a shared problem.
- Coordinate with jschmied (Zig CUDA server, PR #443; external drafter PR #455) and MiaAI-Lab (CUDA Flash Next).
