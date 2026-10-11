"""Build immutable official and decontaminated ASAP snapshots."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
from echolens.ingest import read_records, import_records, quality_summary
from echolens.quality import duplicate_groups, clean_membership

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Smoke test only; writes separate smoke artifacts")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/stage1.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "configs/asap.json").read_text(encoding="utf-8"))
    aspects = list(json.loads((ROOT / "configs/labels/asap-v0.1.json").read_text(encoding="utf-8"))["aspects"])
    folder = ROOT / "data/processed" / ("stage1-smoke" if args.limit else "stage1")
    folder.mkdir(parents=True, exist_ok=True)
    records, imports = [], {}
    for split in ("train", "dev", "test"):
        path = ROOT / "data/raw/asap" / source["revision"] / f"data/{split}.csv"
        rows, stats = read_records(path, mapping={"source_record_id": "id", "text_raw": "review", "rating": "star"}, source="asap", revision=source["revision"], rights_status="upstream_apache_2.0", split=split, aspects=aspects)
        assert stats["input_rows"] == source["expected_rows"][split] and not stats["rejected"]
        imports[split] = stats
        records.extend(rows[:args.limit] if args.limit else rows)
        if args.limit:
            break
    records.sort(key=lambda r: r["record_id"])
    connection = duckdb.connect(str(folder / "feedback.duckdb"))
    inserted = import_records(connection, records)
    assert import_records(connection, records) == 0
    assert connection.execute("SELECT count(*) FROM feedback").fetchone()[0] == len(records)
    groups, pairs = duplicate_groups(records, config["near_duplicates"], config["seed"])
    clean_ids = clean_membership(records, groups)
    members = defaultdict(list)
    for r in records:
        r["duplicate_group_id"] = groups[r["record_id"]]
        r["clean_keep"] = r["record_id"] in clean_ids
        r["clean_exclusion"] = None if r["clean_keep"] else "duplicate_group_lower_priority"
        members[r["duplicate_group_id"]].append(r)
    snapshot = folder / "records.jsonl"
    snapshot.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    connection.execute("CREATE OR REPLACE TABLE prepared AS SELECT * FROM read_json_auto(?)", [str(snapshot)])
    for version in ("official", "clean"):
        for split in ("train", "dev", "test"):
            query = "SELECT * FROM prepared WHERE split = ?" + (" AND clean_keep" if version == "clean" else "") + " ORDER BY record_id"
            connection.sql(query, params=[split]).write_parquet(str(folder / f"{version}-{split}.parquet"), compression="zstd")
    # Assert all retained groups occur once, stronger than merely no cross-split overlap.
    assert len({r["duplicate_group_id"] for r in records if r["clean_keep"]}) == len(clean_ids)
    report = {"processing_version": config["processing_version"], "source_revision": source["revision"],
              "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(), "aspects": aspects,
              "import": imports, "first_pass_inserted": inserted, "repeat_inserted": 0,
              "quality": quality_summary(records), "near_duplicate_config": config["near_duplicates"],
              "duplicate_pairs": pairs, "duplicate_groups": sum(len(v) > 1 for v in members.values()),
              "cross_split_groups": sum(len({r["split"] for r in v}) > 1 for v in members.values()),
              "groups_with_conflicting_labels": sum(len({tuple(r["labels"]) for r in v}) > 1 for v in members.values()),
              "official_rows": dict(Counter(r["split"] for r in records)),
              "clean_rows": dict(Counter(r["split"] for r in records if r["clean_keep"])),
              "exclusions": [{"source_id": r["source_record_id"], "split": r["split"], "group": r["duplicate_group_id"], "reason": r["clean_exclusion"]} for r in records if not r["clean_keep"]],
              "clean_group_overlap": 0,
              "limitations": ["MinHash candidate search can miss lexical near duplicates; not semantic deduplication", "Transitive duplicate components; not all members necessarily pairwise above threshold", "Priority test > dev > train, then lexical source ID; official files remain unchanged", "No timestamp/product/user metadata; cannot test temporal or user leakage"]}
    (folder / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    public = ROOT / "docs/reports" / ("stage1-data-smoke.json" if args.limit else "stage1-data.json")
    public.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    connection.close()
    print(json.dumps({k: report[k] for k in ("official_rows", "clean_rows", "duplicate_groups", "cross_split_groups", "repeat_inserted", "snapshot_sha256")}, indent=2))


if __name__ == "__main__":
    main()
