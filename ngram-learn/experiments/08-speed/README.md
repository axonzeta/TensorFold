# 08 — Speed of the training pass

Baseline (RC1 build, batched, served capture; `chat_edit.py --only 2`, time split added): a fact is ~72% training
passes (~4.4 s a pass of ~100 rows), ~23% greedy checks (~1.4 s a step), the rest lesson setup.

## Packed-int4 expert products (tried, reverted)

Routed experts were dequantised to bf16 every pass (~300 a layer, 9.8 MB each) to multiply a handful of rows; two
kernels (`fn_train_int4_mm`, `fn_train_int4_mm_t`) multiplied straight from the packed GPTQ words instead. Checked
against the dequantised path (TF_INT4_CHECK=1: forward |diff| ~1e-3 of a ~3–7 scale, backward ~1e-4 of ~0.05, the
dequantised path rounding the weights to bf16), and the facts learned identically (1.00 / 1.00 / 1.00). But the pass
was no faster (4.6 s; the first backward kernel, one thread per k over every n, was 11.5 s until split over n).

## Where the pass goes (TF_TRAIN_PROF=1, 98 rows)

| Part | dequantised experts | int4 experts |
| --- | ---: | ---: |
| forward: DeltaNet / attention / MoE | 0.10 / 0.01 / 0.63 s | 0.11 / 0.01 / 0.62 s |
| backward: **DeltaNet** / attention / MoE | **2.92** / 0.02 / 0.36 s | 2.90 / 0.02 / 0.57 s |
| head and mixer | 0.01 s | 0.01 s |

The DeltaNet backward is 70% of the pass. MoE was the cost before served capture removed the training forward; it is
under a quarter now, so the int4 kernels were reverted. The delta rule's backward kernel (one block a head, one
thread a value column, rows walked backwards) summed over all 128 columns at every row with shared-memory atomics
(128 threads on the same 128 addresses): that is what to fix.

## The delta rule's backward, split (`88e4bad`)

The recurrence is column-local: column j of the adjoint dS depends only on itself and scalars of column j. So one
kernel carries each column back through the rows with no sums across columns (the adjoint in registers) and writes
dS_t, delta_t, dm_t and per-column shares; a second, parallel over (row, head), computes dq = S_t dO,
dk = dS_t delta + e^g S_{t-1} dm, dbeta and dg. Checked against fp64 PyTorch (`train_op_check.sh`: every delta-rule
output within ~1e-7 relative).

| 98 rows | before | after |
| --- | ---: | ---: |
| DeltaNet backward | 2.92 s | **0.41 s** |
| the pass | 4.1 s | **1.6 s** |

## Greedy checks gated on the loss

The per-step greedy check (does every answer come out greedily?) cannot pass while the loss is high; it now runs once
the loss per answer token is below 0.25, and every 5th step (`--check-below`, chat_edit.py and learnd.py).

## A fact, end to end (`chat_edit.py --only 2`, served capture, batched)

| | Project Kestrel | Ana Petrović |
| --- | ---: | ---: |
| RC1 | 124 s (21 steps) | 187 s (30 steps) |
| delta backward split | 97 s (29 steps) | 107 s (30 steps) |
| **+ gated checks** | **71 s** (29 steps) | **77 s** (30 steps) |

Both facts 1.00 recall, 1.00 held-out, neighbours unchanged in every run. What is left: the training pass (1.8 s a
step; MoE forward and backward ~1.0 s of it, DeltaNet ~0.5 s) and ~28 steps a fact.
