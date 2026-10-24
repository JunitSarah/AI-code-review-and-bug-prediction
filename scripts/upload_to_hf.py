"""Upload the trained model folder to the Hugging Face Hub so Render can download it.

  pip install huggingface_hub
  huggingface-cli login            # or: set HF_TOKEN=hf_xxx
  python scripts/upload_to_hf.py artifacts/codebert-defect yourname/codebert-defect [--private]

Then set MODEL_PATH=yourname/codebert-defect on Render (and HF_TOKEN too if the repo is private).
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("repo_id")
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

required = ["config.json", "tokenizer_config.json"]
missing = [f for f in required if not (args.folder / f).exists()]
weights = list(args.folder.glob("*.safetensors")) + list(args.folder.glob("pytorch_model.bin"))
if missing or not weights:
    raise SystemExit(f"{args.folder} doesn't look like a saved model (missing: {missing or 'weights'})")

api = HfApi()
api.create_repo(args.repo_id, private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(args.folder), repo_id=args.repo_id,
                  ignore_patterns=["checkpoints/*", "checkpoint-*", "*.pt", "optimizer*"])
print(f"Uploaded. Use MODEL_PATH={args.repo_id}")
