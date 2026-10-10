# 02b — Forward-only fact editing (SPSA on the overlay)

`tools/zo_edit.py` through `tools/worker.py` (`tensorfold worker`: model loaded once, 54 ms a scored case). CounterFact
counterfactual edits (true → new), first 10 cases. Training prompts: the rewrite statement and the first paraphrase;
held out: the second paraphrase; neighbourhood: 5 prompts about other subjects. Parameters: subject-position rows,
moved in a 64-dimensional random subspace scaled to the rows' size; antithetic SPSA + Adam; init from a transplant
(rows of a subject whose true answer is the target), objective log p(new) − log p(true).

| Run | Rows | Efficacy | Generalisation | Specificity | s / fact |
| --- | --- | ---: | ---: | ---: | ---: |
| before | — | 0.20 | 0.10 | 0.86 | |
| transplant only | all subject rows | 0.17 | 0.22 | 0.84 | |
| pilot 1 (log p(new), no init; 1 case) | all | 0/1 | 0/1 | 1/1 | 121 |
| pilot 2 | all subject rows (16–104) | 0.50 | 0.20 | 0.86 | 50 |
| pilot 3 | context-free rows only, rarity-weighted | 0.55 | 0.30 | 0.86 | 49 |

Reading: forward-only search edits about half the training prompts and never disturbs the neighbourhood, but it
generalises poorly and is slow per fact; it is not a substitute for gradients. Context-free rows (inside the subject)
help slightly. Next: gradients (PyTorch reference trainer, then in-engine backward).
