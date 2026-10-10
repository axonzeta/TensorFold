# patches/

The engine changes as they were developed: each file here is a copy of a file in the fork, and each `apply_*.py`
patched the base checkout (TensorFold `zig-flashnext` `db28187` with MiaAI-Lab's patches 0001-0009) in place.
They are kept as the development record. **The branch already contains every change as commits** (`git log
db28187..ngram-learn`); nothing here needs applying to build it.

| Script | What it added |
| --- | --- |
| `apply_overlay.py` | `TF_NGRAM_OVERLAY`, the overlay in the host n-gram gather |
| `apply_probe.py`, `apply_worker.py` | `tensorfold probe`, `score`, `worker` (the experiments' JSON-lines engine) |
| `apply_train_kernels.py`, `apply_train_check.py` | `fn_train.cu` and the per-op checks |
| `apply_dequant.py`, `apply_train_model.py` | weights as bf16 from the engine's buffers; the training pass |
| `apply_capture.py` | the served prefill's capture hook |
| `apply_serve_learn.py` | learn ops in the server, overlays by tag (Phase 4) |
