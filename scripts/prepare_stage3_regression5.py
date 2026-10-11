"""Freeze five known train cases for a narrow rule regression; do not annotate."""
import json

from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json
from check_stage3_batch import load_task

PRIOR = ROOT / "data/annotations/stage3-teacher-next80-v3"
OUT = ROOT / "data/annotations/stage3-regression5-v3.1"
CASES = [("41904", "batch-0011"), ("22886", "batch-0015"),
         ("17987", "batch-0016"), ("17571", "batch-0015"), ("27257", "batch-0020")]


def main():
    if (OUT / "manifest.json").exists():
        load_task(OUT)
        print("Existing regression5 v3.1 preserved and hashes verified")
        return
    parent = load_task(PRIOR)
    labels = PRIOR / "labels.json"
    aspects = read_json(labels)["aspects"]
    assert len(aspects) == 28
    copies = {"rules.md": ROOT / "docs/annotation-teacher-v3.1.md",
              "labels.json": labels, "START_HERE.md": ROOT / "docs/stage3-codex-regression5-task.md"}
    prepared = []
    for source_id, parent_batch in CASES:
        source = PRIOR / "inputs" / f"{parent_batch}.json"
        payload = read_json(source)
        matches = [row for row in payload["records"] if str(row["source_record_id"]) == source_id]
        assert len(matches) == 1
        row = matches[0]
        assert set(row) == {"record_id", "source_record_id", "text_raw", "spans"}
        assert "".join(s["text"] for s in row["spans"]) == row["text_raw"]
        assert payload["aspects"] == aspects
        meta = parent["batches"][parent_batch]
        index = meta["record_ids"].index(row["record_id"])
        prepared.append((row, source, meta["duplicate_group_ids"][index]))
    assert len({row["record_id"] for row, _, _ in prepared}) == 5
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in copies.items():
        (OUT / name).write_bytes(source.read_bytes())
    for name in ("outputs", "checks"):
        (OUT / name).mkdir(exist_ok=True)
    files, batches = list(copies), {}
    for index, (row, source, group) in enumerate(prepared, 1):
        batch = f"batch-{index:04d}"
        name = f"inputs/{batch}.json"
        atomic_json(OUT / name, {"schema_version": "teacher-coverage-v1",
                                "guideline_version": "teacher-v3.1", "aspects": aspects, "records": [row]})
        files.append(name)
        batches[batch] = {"record_ids": [row["record_id"]], "source_record_ids": [row["source_record_id"]],
                          "duplicate_group_ids": [group], "characters": len(row["text_raw"]),
                          "parent_batch": source.stem, "parent_input_sha256": sha256(source)}
    manifest = {"records": 5, "batch_size": 1, "guideline_version": "teacher-v3.1",
                "label_version": "teacher-v3", "schema_version": "teacher-coverage-v1",
                "requested_model": "gpt-5.6-sol", "thinking": "unknown until declared",
                "source_split": "clean-train", "parent_task": str(PRIOR.relative_to(ROOT)),
                "parent_manifest_sha256": sha256(PRIOR / "manifest.json"),
                "source_train_sha256": parent["source_train_sha256"], "batches": batches,
                "usage": "Known-case development regression; not unseen evaluation or human gold",
                "origin": "teacher_model_unreviewed", "human_gold": False, "training_ready": False,
                "files": {name: sha256(OUT / name) for name in files}}
    atomic_json(OUT / "manifest.json", manifest)
    report = {"records": 5, "batches": 5, "batch_size": 1,
              "source_record_ids": [source_id for source_id, _ in CASES],
              "characters": sum(len(row["text_raw"]) for row, _, _ in prepared),
              "unchanged_original_text_and_spans": True, "aspect_count": len(aspects),
              "unchanged_label_sha256": sha256(labels), "guideline_version": "teacher-v3.1",
              "known_case_regression": True, "unseen_evaluation": False, "prepared_only": True,
              "model_calls": 0, "human_gold": False, "training_ready": False,
              "manifest_sha256": sha256(OUT / "manifest.json")}
    atomic_json(ROOT / "docs/reports/stage3-regression5-v3.1-preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
