# References

Numbers as reported by each paper; items marked † were cited from memory by the research pass and not re-fetched.

## Editing n-gram / memory tables

- **EngramEdit** — Cai, Wei, Wang, Wu, Song, Li, Li. arXiv 2610.10533, 7 Oct 2026. https://arxiv.org/abs/2610.10533
  Edits n-gram embeddings active at the last subject token (found by causal tracing), not the answer position. One
  shared perturbation over K=4 generated paraphrases, then a joint solve for embedding updates across expressions and
  edits via an expression→embedding map, penalising short / frequent n-grams. Base: LongCat-Flash-Lite; names
  Qwen3.8-Flash-Next as Engram-style. CounterFact 2k edits Eff/Gen/Spec 99.5/97.0/85.2 (FT 89.4/60.9/75.6);
  >96% of pre-edit F1 after 5k sequential edits. Paraphrases lift generalisation 75.0 → 97.0. 2-grams generalise best,
  longer n-grams localise best. >92% of new errors on unrelated queries involve an updated embedding.
- **Continual Learning via Sparse Memory Finetuning** — Lin, Zettlemoyer, Ghosh, Yih, Markosyan, Berges, Oğuz.
  arXiv 2510.15103, 2025. https://arxiv.org/abs/2510.15103
  Updates only top-t memory slots by TF-IDF against 1,000 background batches. NaturalQuestions F1 drop at equal
  acquisition: full FT 89%, LoRA 71%, sparse memory FT 11%.

## Engram and memory layers

- **Conditional Memory via Scalable Lookup (Engram)** — Cheng, Zeng, Dai et al., arXiv 2601.07372, Jan 2026.
  https://arxiv.org/abs/2601.07372 — suppressing Engram leaves factual tasks at 29–44%, reading comprehension 81–93%.
- **Engram-Nine** — Tao Lin, arXiv 2601.16531, 2026: collision-free hashing gave no gain; collisions act as
  regularisation; the gate favours frequent n-grams (small scale).
- **Memory Layers at Scale** — Berges et al., arXiv 2412.09764, 2024.
- **Scaling Embeddings Outperforms Scaling Experts (LongCat-Flash-Lite)** — arXiv 2601.21204, 2026.
- Product Key Memory (Lample et al., 2019, arXiv 1907.05242)†; UltraMem (arXiv 2411.12364); PEER (arXiv 2407.04153)†.

## Knowledge editing and its failure modes

- ROME (arXiv 2202.05262), MEMIT (arXiv 2210.07229): edit at the last subject token.
- Dissecting Recall of Factual Associations — Geva et al., arXiv 2304.14767: attention carries subject information to
  the answer position.
- Model editing at scale leads to gradual and catastrophic forgetting — Gupta et al., arXiv 2401.07453: ROME collapses
  after 100–1000 sequential edits, MEMIT ~1.5k–4k.
- WISE (arXiv 2405.14768): side memory; GRACE (arXiv 2211.11031) stays local but paraphrase generalisation falls to
  0.08 at 1k edits — the risk for any purely key-matched memory.
- MELO†, MEMOIR†, AlphaEdit†, RippleEdits†, MQuAKE†.
- Forgetting Is Not a Fix — Schessl, arXiv 2607.24805: sequential edits are order-dependent (may concern a different
  "engram"; unverified).

## Forward-only training

- MeZO — Malladi et al., NeurIPS 2023 (arXiv 2305.17333)†: fine-tuning with forward passes only.

## TensorFold

- Living Weights guide: `TensorFold/docs/living-weights.md` (upstream `f1aef9c`).
- MiaAI-Lab Zig CUDA Flash Next (MIT): https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
- External drafter interface (draft PR): https://github.com/ashhart/TensorFold/pull/455
- woct0rdho Qwen recipe (full backward of qwen4_exp on PyTorch/ROCm):
  https://github.com/woct0rdho/transformers5-qwen3.5-recipe
