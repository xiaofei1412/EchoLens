"""Fetch pinned model artifacts into one local copy."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
for model in json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8")).values():
    print(snapshot_download(model["repo_id"], revision=model["revision"], local_dir=ROOT / model["local_dir"], max_workers=2))
