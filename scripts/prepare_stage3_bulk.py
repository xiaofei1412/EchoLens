"""Freeze 900 unseen train-pool candidates for local teacher annotation."""
import json
import random

import duckdb

from echolens.annotation import load_packet
from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json
from prepare_stage3_teacher import segments
from check_stage3_batch import TASK, load_task

SEED = 2026100903


def main():
    if (TASK / "manifest.json").exists():
        load_task(TASK)
        print("Existing 900-record task preserved; frozen files verified")
        return
    fresh = ROOT / "data/annotations/stage3-teacher-fresh10-v1"
    prior = read_json(fresh / "manifest.json")
    assert all(sha256(fresh / name) == h for name, h in prior["files"].items())
    exposed = set(prior["excluded_source_ids"])
    exposed.update(str(r["source_record_id"]) for r in read_json(fresh / "inputs.json")["records"])
    train = ROOT / "data/processed/stage1/clean-train.parquet"
    with duckdb.connect() as db:
        cursor = db.execute("SELECT record_id,source_record_id,text_raw,text_normalized,duplicate_group_id,split,clean_keep,rights_status FROM read_parquet(?) ORDER BY record_id", [str(train)])
        names = [c[0] for c in cursor.description]
        rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    assert all(r["split"] == "train" and r["clean_keep"] for r in rows)
    old = [r for r in rows if str(r["source_record_id"]) in exposed]
    groups, texts = {r["duplicate_group_id"] for r in old}, {r["text_normalized"] for r in old}
    eligible = [r for r in rows if str(r["source_record_id"]) not in exposed
                and r["duplicate_group_id"] not in groups and r["text_normalized"] not in texts
                and r["rights_status"] == "upstream_apache_2.0"]
    rng = random.Random(SEED)
    first, rest, seen_groups, seen_texts = [], [], set(), set()
    for low, high, count, initial in ((0, 200, 90, 4), (200, 500, 540, 10), (500, 10000, 270, 6)):
        pool = [r for r in eligible if low < len(r["text_raw"]) <= high]
        rng.shuffle(pool)
        chosen = []
        for row in pool:
            if row["duplicate_group_id"] in seen_groups or row["text_normalized"] in seen_texts:
                continue
            chosen.append(row)
            seen_groups.add(row["duplicate_group_id"])
            seen_texts.add(row["text_normalized"])
            if len(chosen) == count:
                break
        assert len(chosen) == count
        first.extend(chosen[:initial])
        rest.extend(chosen[initial:])
    rng.shuffle(first)
    rng.shuffle(rest)
    selected = first + rest
    assert len(selected) == len(seen_groups) == len(seen_texts) == 900
    aspects = load_packet()["aspects"]
    TASK.mkdir(parents=True, exist_ok=True)
    rules = ROOT / "docs/annotation-teacher-v2.md"
    (TASK / "rules.md").write_bytes(rules.read_bytes())
    start = ROOT / "docs/stage3-codex-annotation-task.md"
    (TASK / "START_HERE.md").write_bytes(start.read_bytes())
    for directory in ("outputs", "checks"):
        (TASK / directory).mkdir(exist_ok=True)
    files, batches = ["rules.md", "START_HERE.md"], {}
    for index in range(0, 900, 5):
        batch = f"batch-{index//5+1:04d}"
        subset = selected[index:index+5]
        inputs = {"schema_version": "teacher-coverage-v1", "guideline_version": "teacher-v2",
                  "aspects": aspects, "records": [{"record_id": r["record_id"], "source_record_id": r["source_record_id"],
                  "text_raw": r["text_raw"], "spans": segments(r["text_raw"])} for r in subset]}
        assert all("".join(s["text"] for s in r["spans"]) == r["text_raw"] for r in inputs["records"])
        name = f"inputs/{batch}.json"
        atomic_json(TASK / name, inputs)
        files.append(name)
        batches[batch] = {"records": 5, "record_ids": [r["record_id"] for r in subset],
                          "duplicate_group_ids": [r["duplicate_group_id"] for r in subset],
                          "characters": sum(len(r["text_raw"]) for r in subset),
                          "wave": "rule_validation_20" if index < 20 else "after_rule_review"}
    manifest = {"seed": SEED, "records": 900, "batch_size": 5, "requested_model": "gpt-5.6-sol",
                "thinking": "unknown; record actual setting when known", "source_split": "clean-train",
                "source_train_sha256": sha256(train), "prior_manifest_sha256": sha256(fresh / "manifest.json"),
                "excluded_source_ids": sorted(exposed), "length_strata": {"1-200": 90, "201-500": 540, "501+": 270},
                "pilot_length_strata": {"1-200": 4, "201-500": 10, "501+": 6}, "batches": batches,
                "origin": "teacher_model_unreviewed", "training_ready": False, "human_gold": False,
                "usage": "Train-only candidates; not 900 human gold labels; no dev/test accessed or frozen here",
                "files": {name: sha256(TASK / name) for name in files}}
    atomic_json(TASK / "manifest.json", manifest)
    report = {"records": 900, "batches": len(batches), "seed": SEED, "excluded_source_ids": len(exposed),
              "overlap_with_prior_exposure": 0, "distinct_duplicate_groups": len(seen_groups),
              "length_strata": manifest["length_strata"], "pilot_length_strata": manifest["pilot_length_strata"],
              "max_batch_characters": max(b["characters"] for b in batches.values()),
              "manifest_sha256": sha256(TASK / "manifest.json"), "rule_sha256": sha256(rules),
              "prepared_only": True, "model_calls": 0, "human_gold": False, "training_ready": False}
    atomic_json(ROOT / "docs/reports/stage3-bulk-preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
