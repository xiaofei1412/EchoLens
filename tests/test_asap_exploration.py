"""Meaningful local-data checks; skipped on a code-only checkout."""
import csv
import hashlib
import json
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_downloads_match_recorded_hashes():
    config = json.loads((ROOT / "configs/asap.json").read_text(encoding="utf-8"))
    directory = ROOT / "data/raw/asap" / config["revision"]
    if not directory.exists():
        pytest.skip("Download ASAP first")
    manifest = json.loads((ROOT / "docs/reports/asap-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert hashlib.sha256((directory / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_exploration_is_train_only_and_disjoint_from_held_out_text():
    config = json.loads((ROOT / "configs/asap.json").read_text(encoding="utf-8"))
    source = ROOT / "data/raw/asap" / config["revision"] / "data"
    sample_file = ROOT / "data/exploration/asap-200.jsonl"
    if not sample_file.exists():
        pytest.skip("Run fetch_asap.py and audit_asap.py first")
    samples = [json.loads(line) for line in sample_file.read_text(encoding="utf-8").splitlines()]
    def read(split):
        with (source / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    train = {r["id"]: r["review"] for r in read("train")}
    held_out = {re.sub(r"\s+", "", r["review"]) for split in ("dev", "test") for r in read(split)}
    assert len(samples) == len({s["source_id"] for s in samples}) == 200
    for sample in samples:
        assert sample["split"] == "train"
        assert train[sample["source_id"]] == sample["text"]
        assert re.sub(r"\s+", "", sample["text"]) not in held_out
