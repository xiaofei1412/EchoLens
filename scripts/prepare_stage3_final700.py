"""Freeze the user-authorized final 700 candidates; preserve prior tasks."""
import json

import duckdb

from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json
from check_stage3_batch import TASK as PRIOR, load_task

OUT = ROOT / "data/annotations/stage3-teacher-final700-v3.2"
WAVE = ROOT / "data/annotations/stage3-teacher-wave100-v3.2"


def main():
    if (OUT / "manifest.json").exists():
        load_task(OUT)
        print("Existing final700 v3.2 preserved and frozen hashes verified")
        return
    parent = load_task(PRIOR)
    wave = load_task(WAVE)
    assert parent["records"] == 900 and wave["records"] == 100
    train = ROOT / "data/processed/stage1/clean-train.parquet"
    assert sha256(train) == parent["source_train_sha256"]
    with duckdb.connect() as db:
        cursor = db.execute("SELECT record_id,source_record_id,text_raw,text_normalized,duplicate_group_id,split,clean_keep,rights_status FROM read_parquet(?)", [str(train)])
        names = [c[0] for c in cursor.description]
        rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    by_id = {r["record_id"]: r for r in rows}
    earlier = {rid for n in range(1, 41) for rid in parent["batches"][f"batch-{n:04d}"]["record_ids"]}
    wave_ids = {rid for b in wave["batches"].values() for rid in b["record_ids"]}
    expected_wave = {rid for n in range(21, 41) for rid in parent["batches"][f"batch-{n:04d}"]["record_ids"]}
    assert wave_ids == expected_wave and len(earlier) == 200
    excluded = set(parent["excluded_source_ids"])
    old_rows = [r for r in rows if r["record_id"] in earlier or str(r["source_record_id"]) in excluded]
    old_groups = {r["duplicate_group_id"] for r in old_rows}
    old_texts = {r["text_normalized"] for r in old_rows}
    selected = []
    for n in range(41, 181):
        batch = f"batch-{n:04d}"
        assert not list((PRIOR / "outputs").glob(f"{batch}.r*.json"))
        assert not list((PRIOR / "checks").glob(f"{batch}.r*.json"))
        source = PRIOR / "inputs" / f"{batch}.json"
        meta = parent["batches"][batch]
        for row in read_json(source)["records"]:
            assert set(row) == {"record_id", "source_record_id", "text_raw", "spans"}
            original = by_id[row["record_id"]]
            group = meta["duplicate_group_ids"][meta["record_ids"].index(row["record_id"])]
            assert original["split"] == "train" and original["clean_keep"]
            assert original["rights_status"] == "upstream_apache_2.0"
            assert original["text_raw"] == row["text_raw"]
            assert str(original["source_record_id"]) == str(row["source_record_id"])
            assert original["duplicate_group_id"] == group
            assert row["record_id"] not in earlier and str(row["source_record_id"]) not in excluded
            assert group not in old_groups and original["text_normalized"] not in old_texts
            assert "".join(s["text"] for s in row["spans"]) == row["text_raw"]
            selected.append((row, group, original["text_normalized"], source))
    assert len(selected) == len({r[0]["record_id"] for r in selected}) == 700
    assert len({r[1] for r in selected}) == len({r[2] for r in selected}) == 700
    all_parent_ids = {rid for b in parent["batches"].values() for rid in b["record_ids"]}
    assert earlier | {r[0]["record_id"] for r in selected} == all_parent_ids
    copies = {"rules.md": WAVE / "rules.md", "labels.json": WAVE / "labels.json",
              "START_HERE.md": ROOT / "docs/stage3-codex-final700-task.md"}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in copies.items():
        (OUT / name).write_bytes(source.read_bytes())
    for name in ("outputs", "checks"):
        (OUT / name).mkdir(exist_ok=True)
    aspects = read_json(OUT / "labels.json")["aspects"]
    files, batches, lengths = list(copies), {}, []
    for n, (row, group, _, source) in enumerate(selected, 1):
        batch = f"batch-{n:04d}"
        name = f"inputs/{batch}.json"
        atomic_json(OUT / name, {"schema_version": "teacher-coverage-v1", "guideline_version": "teacher-v3.2",
                                "aspects": aspects, "records": [row]})
        files.append(name)
        lengths.append(len(row["text_raw"]))
        batches[batch] = {"record_ids": [row["record_id"]], "source_record_ids": [row["source_record_id"]],
                          "duplicate_group_ids": [group], "characters": len(row["text_raw"]),
                          "progress_section": (n - 1) // 100 + 1,
                          "parent_batch": source.stem, "parent_input_sha256": sha256(source)}
    manifest = {"records": 700, "batch_size": 1, "guideline_version": "teacher-v3.2", "label_version": "teacher-v3",
                "schema_version": "teacher-coverage-v1", "requested_model": "gpt-5.6-sol",
                "thinking": "unknown until declared", "source_split": "clean-train",
                "parent_task": str(PRIOR.relative_to(ROOT)), "parent_manifest_sha256": sha256(PRIOR / "manifest.json"),
                "source_train_sha256": parent["source_train_sha256"], "parent_seed": parent["seed"],
                "excluded_source_ids": parent["excluded_source_ids"], "batches": batches,
                "preceding_wave_manifest_sha256": sha256(WAVE / "manifest.json"),
                "authorization": "User accepted current quality; no further wave100 review or repair prerequisite; prepare all remaining 700",
                "usage": "Final 700 of original 900 train candidates; AI annotation, not human gold or test",
                "origin": "teacher_model_unreviewed", "human_gold": False, "training_ready": False,
                "files": {name: sha256(OUT / name) for name in files}}
    atomic_json(OUT / "manifest.json", manifest)
    report = {"records": 700, "batches": 700, "batch_size": 1, "progress_sections": 7,
              "parent_batches": "0041-0180", "source_split": "clean-train", "prior_records": 200,
              "prior200_and_excluded_record_group_normalized_text_overlap": 0,
              "excluded_source_ids": len(excluded), "distinct_duplicate_groups": 700,
              "distinct_normalized_texts": 700, "original900_exact_partition": True,
              "unchanged_original_text_and_spans": True, "aspect_count": len(aspects),
              "same_rules_and_labels_as_wave100": True,
              "length_strata": {"1-200": sum(n <= 200 for n in lengths),
                                "201-500": sum(200 < n <= 500 for n in lengths), "501+": sum(n > 500 for n in lengths)},
              "prepared_only": True, "model_calls": 0, "human_gold": False, "training_ready": False,
              "remaining_after_this_task": 0, "manifest_sha256": sha256(OUT / "manifest.json")}
    atomic_json(ROOT / "docs/reports/stage3-final700-v3.2-preparation.json", report)
    load_task(OUT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
