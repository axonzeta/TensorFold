"""The background corpus for tools/rowfreq.py (experiments/01b-row-frequency): FineWeb sample-10BT shards 000-001,
UltraChat 200k SFT, codeparrot-clean files 1-4, into data/bg."""
from huggingface_hub import hf_hub_download
import pathlib
D=str(pathlib.Path(__file__).resolve().parent / "bg")
for f in ["sample/10BT/000_00000.parquet","sample/10BT/001_00000.parquet"]: hf_hub_download("HuggingFaceFW/fineweb",f,repo_type="dataset",local_dir=D+"/fineweb")
for f in ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet","data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet","data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"]: hf_hub_download("HuggingFaceH4/ultrachat_200k",f,repo_type="dataset",local_dir=D+"/ultrachat")
for i in range(1,5): hf_hub_download("codeparrot/codeparrot-clean",f"file-{i:012d}.json.gz",repo_type="dataset",local_dir=D+"/code")
print("FETCHED")
