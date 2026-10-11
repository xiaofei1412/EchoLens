"""Create a v2 dataset with identical input rows; never mutate the v1 baseline."""
import random
import shutil
from echolens.experiments import ROOT, sha256
from echolens.workbench import dataset_path, records_at, read_json, save_dataset, atomic_json


def main():
    source, target = dataset_path("asap-demo-1000"), dataset_path("asap-demo-1000-v2")
    if (target / "dataset.json").exists():
        print("Existing v2 dataset preserved")
        return
    rows = records_at(source)
    metadata = {"name":"ASAP · 1,000 条 · 阶段2修订 v2", "kind":"train_demo",
                "analysis_config":"configs/stage2-v2.json", "baseline_dataset":source.name}
    save_dataset(target, rows, metadata)
    assert sha256(source/"records.jsonl") == sha256(target/"records.jsonl")
    shutil.copy2(source/"classifications.json", target/"classifications.json")
    cases = {"5365", "38689", "6479", "44572", "32524"}
    required = [r["record_id"] for r in rows if r["source_record_id"] in cases]
    remaining = [r["record_id"] for r in rows if r["record_id"] not in required]
    random.Random(20261010).shuffle(remaining)
    atomic_json(target/"audit-selection.json", {"purpose":"Train-only development review, not gold or held-out evaluation",
                "user_reported":required, "pilot":required+remaining[:27], "additional_audit":remaining[27:59]})
    atomic_json(target/"baseline-preservation.json", {p.name:sha256(p) for p in [source/"records.jsonl",source/"classifications.json",source/"topics.json"]})
    print(f"Prepared {len(rows)} unchanged records and 32-row pilot; classifier predictions reused unchanged")


if __name__ == "__main__":
    main()
