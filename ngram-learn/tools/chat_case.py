"""Write the repro's chat lesson as a train-grad-check case.json: python chat_case.py DIR"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

M = Path("/hf/hub/models--azampatti--Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound/snapshots/1464274120d36a4d8fcaa934552334a7d83ce0fd")
tok = Tokenizer.from_file(str(M / "tokenizer.json"))
p = tok.encode("<|im_start|>user\nWhen does Project Kestrel launch?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
a = tok.encode("Project Kestrel launches on 3 November.<|im_end|>", add_special_tokens=False).ids
toks = p + a[:-1]
d = Path(sys.argv[1])
d.mkdir(parents=True, exist_ok=True)
(d / "case.json").write_text(json.dumps({"tokens": toks, "at": list(range(len(p) - 1, len(toks))), "targets": a}))
print(len(toks), "rows")
