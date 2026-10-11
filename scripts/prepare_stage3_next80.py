"""Version the next 80 previously prepared train records; do not annotate them."""
import json

from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json
from check_stage3_batch import TASK as PRIOR, load_task

OUT = ROOT / "data/annotations/stage3-teacher-next80-v3"


def main():
    if (OUT / "manifest.json").exists():
        load_task(OUT)
        print("Existing next80 v3 preserved and hashes verified")
        return
    old = load_task(PRIOR)
    labels_path = ROOT / "configs/labels/teacher-v3.json"
    aspects = read_json(labels_path)["aspects"]
    copies = {"rules.md": ROOT / "docs/annotation-teacher-v3.md",
              "labels.json": labels_path, "START_HERE.md": ROOT / "docs/stage3-codex-next80-task.md"}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in copies.items():
        (OUT / name).write_bytes(source.read_bytes())
    for name in ("outputs", "checks"):
        (OUT / name).mkdir(exist_ok=True)
    first_ids = {rid for b in list(old["batches"])[:4] for rid in old["batches"][b]["record_ids"]}
    first_groups = {g for b in list(old["batches"])[:4] for g in old["batches"][b]["duplicate_group_ids"]}
    files, batches, lengths, selected_ids, selected_groups = list(copies), {}, [], set(), set()
    for index in range(5, 21):
        batch = f"batch-{index:04d}"
        if list((PRIOR / "outputs").glob(f"{batch}.r*.json")) or list((PRIOR / "checks").glob(f"{batch}.r*.json")):
            raise ValueError(f"{batch} already has outputs; cannot describe it as an unannotated validation batch")
        source = PRIOR / "inputs" / f"{batch}.json"
        payload = read_json(source)
        payload.update(guideline_version="teacher-v3", aspects=aspects)
        for row in payload["records"]:
            assert set(row) == {"record_id", "source_record_id", "text_raw", "spans"}
            assert "".join(s["text"] for s in row["spans"]) == row["text_raw"]
            assert row["record_id"] not in selected_ids | first_ids
            selected_ids.add(row["record_id"])
            lengths.append(len(row["text_raw"]))
        group_ids = old["batches"][batch]["duplicate_group_ids"]
        assert not set(group_ids) & (selected_groups | first_groups)
        selected_groups.update(group_ids)
        name = f"inputs/{batch}.json"
        atomic_json(OUT / name, payload)
        files.append(name)
        batches[batch] = {**old["batches"][batch], "wave": "next80_v3", "parent_input_sha256": sha256(source)}
    assert len(selected_ids) == len(selected_groups) == 80
    manifest = {"records": 80, "batch_size": 5, "guideline_version": "teacher-v3", "schema_version": "teacher-coverage-v1",
                "requested_model": "gpt-5.6-sol", "thinking": "unknown until declared", "source_split": "clean-train",
                "parent_task": str(PRIOR.relative_to(ROOT)), "parent_manifest_sha256": sha256(PRIOR / "manifest.json"),
                "source_train_sha256": old["source_train_sha256"], "parent_seed": old["seed"],
                "excluded_source_ids": old["excluded_source_ids"], "batches": batches,
                "usage": "Unannotated train-pool development validation; not frozen test or human gold",
                "origin": "teacher_model_unreviewed", "human_gold": False, "training_ready": False,
                "files": {name: sha256(OUT / name) for name in files}}
    atomic_json(OUT / "manifest.json", manifest)
    report = {"records": 80, "batches": 16, "first_batch": "batch-0005", "last_batch": "batch-0020",
              "length_strata": {"1-200": sum(n <= 200 for n in lengths), "201-500": sum(200 < n <= 500 for n in lengths),
                                "501+": sum(n > 500 for n in lengths)},
              "duplicate_groups": 80, "prior20_id_group_overlap": 0, "unchanged_original_text_and_spans": True,
              "aspect_count": len(aspects), "added_aspect_count": 7, "prepared_only": True, "model_calls": 0,
              "human_gold": False, "training_ready": False, "manifest_sha256": sha256(OUT / "manifest.json")}
    atomic_json(ROOT / "docs/reports/stage3-next80-v3-preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
