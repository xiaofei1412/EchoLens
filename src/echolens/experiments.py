"""Shared local snapshot/provenance helpers for stage 1 experiments."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def configuration():
    config = json.loads((ROOT / "configs/stage1.json").read_text(encoding="utf-8"))
    aspects = list(json.loads((ROOT / "configs/labels/asap-v0.1.json").read_text(encoding="utf-8"))["aspects"])
    return config, aspects


def load_split(version, split):
    if version not in ("official", "clean") or split not in ("train", "dev", "test"):
        raise ValueError("Invalid dataset partition")
    path = ROOT / "data/processed/stage1" / f"{version}-{split}.parquet"
    with duckdb.connect() as connection:
        rows = connection.execute("SELECT record_id, text_normalized, labels FROM read_parquet(?) WHERE rights_status = 'upstream_apache_2.0' ORDER BY record_id", [str(path)]).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows], np.asarray([r[2] for r in rows], dtype=np.int64)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance():
    manifest = json.loads((ROOT / "data/processed/stage1/manifest.json").read_text(encoding="utf-8"))
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "duckdb", "datasketch")},
            "snapshot_sha256": manifest["snapshot_sha256"], "config_sha256": sha256(ROOT / "configs/stage1.json")}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
