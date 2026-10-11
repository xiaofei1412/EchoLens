"""Prepare train-only workbench demo without reading held-out splits."""
import json
import random
import duckdb
from echolens.experiments import ROOT
from echolens.workbench import dataset_path, save_dataset


def main():
    config = json.loads((ROOT / "configs/stage2.json").read_text(encoding="utf-8"))
    path = ROOT / "data/processed/stage1/clean-train.parquet"
    with duckdb.connect() as con:
        cursor = con.execute("SELECT * FROM read_parquet(?) ORDER BY record_id", [str(path)])
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
    sample = random.Random(config["seed"]).sample(rows, config["sample_size"])
    assert len(sample) == 1000 and all(r["split"] == "train" and r["clean_keep"] for r in sample)
    folder = dataset_path("asap-demo-1000")
    count = save_dataset(folder, sample, {"name": "ASAP · 1,000 条餐饮反馈", "kind": "train_demo", "seed": config["seed"], "source": "stage1 clean-train", "note": "Training-set workflow demo, not held-out performance evaluation"})
    print(f"Prepared {len(sample)} records, inserted={count}; {folder}")


if __name__ == "__main__":
    main()
