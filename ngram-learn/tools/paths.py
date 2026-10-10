"""Where the checkpoint lives on the host: $HF_DIR (default ~/qwen38/hf), the Hugging Face cache layout under it."""

from __future__ import annotations

import os
from pathlib import Path

HF_DIR = Path(os.environ.get("HF_DIR", Path.home() / "qwen38" / "hf"))
REPO = "models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound"
REVISION = "1464274120d36a4d8fcaa934552334a7d83ce0fd"
MODEL = HF_DIR / "hub" / REPO / "snapshots" / REVISION
