"""Prepare the next planned 100 train candidates, one record per batch."""
import json

from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json
from check_stage3_batch import TASK as PRIOR, load_task

OUT = ROOT / "data/annotations/stage3-teacher-wave100-v3.2"


def main():
    if (OUT / "manifest.json").exists():
        load_task(OUT)
        print("Existing wave100 v3.2 preserved and hashes verified")
        return
    parent = load_task(PRIOR)
    revised = ROOT / "data/annotations/stage3-ai-revised80-v3.2"
    revision = read_json(revised / "final-manifest.json")
    assert all(sha256(revised / n) == h for n,h in revision["files"].items())
    assert revision["release_next_wave_for_reviewed_annotation"] is True
    labels_path = ROOT / "configs/labels/teacher-v3.json"
    aspects = read_json(labels_path)["aspects"]
    earlier = [parent["batches"][f"batch-{n:04d}"] for n in range(1, 21)]
    old_ids = {rid for b in earlier for rid in b["record_ids"]}
    old_groups = {g for b in earlier for g in b["duplicate_group_ids"]}
    selected = []
    for n in range(21, 41):
        batch = f"batch-{n:04d}"
        assert not list((PRIOR / "outputs").glob(f"{batch}.r*.json"))
        assert not list((PRIOR / "checks").glob(f"{batch}.r*.json"))
        source = PRIOR / "inputs" / f"{batch}.json"
        payload = read_json(source)
        meta = parent["batches"][batch]
        for row in payload["records"]:
            assert set(row) == {"record_id", "source_record_id", "text_raw", "spans"}
            group = meta["duplicate_group_ids"][meta["record_ids"].index(row["record_id"])]
            assert row["record_id"] not in old_ids and group not in old_groups
            assert "".join(s["text"] for s in row["spans"]) == row["text_raw"]
            selected.append((row, group, source))
    assert len(selected) == len({row["record_id"] for row,_,_ in selected}) == 100
    assert len({g for _,g,_ in selected}) == 100
    copies = {"rules.md": ROOT / "docs/annotation-teacher-v3.2.md", "labels.json": labels_path,
              "START_HERE.md": ROOT / "docs/stage3-codex-wave100-task.md"}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in copies.items():
        (OUT / name).write_bytes(source.read_bytes())
    for name in ("outputs", "checks"):
        (OUT / name).mkdir(exist_ok=True)
    files, batches, lengths = list(copies), {}, []
    for n,(row,group,source) in enumerate(selected,1):
        b = f"batch-{n:04d}"
        name = f"inputs/{b}.json"
        atomic_json(OUT / name, {"schema_version":"teacher-coverage-v1", "guideline_version":"teacher-v3.2",
                                "aspects":aspects, "records":[row]})
        files.append(name)
        lengths.append(len(row["text_raw"]))
        batches[b] = {"record_ids":[row["record_id"]], "source_record_ids":[row["source_record_id"]],
                      "duplicate_group_ids":[group], "characters":len(row["text_raw"]),
                      "parent_batch":source.stem, "parent_input_sha256":sha256(source)}
    manifest = {"records":100, "batch_size":1, "guideline_version":"teacher-v3.2", "label_version":"teacher-v3",
                "schema_version":"teacher-coverage-v1", "requested_model":"gpt-5.6-sol",
                "thinking":"unknown until declared", "source_split":"clean-train",
                "parent_task":str(PRIOR.relative_to(ROOT)), "parent_manifest_sha256":sha256(PRIOR / "manifest.json"),
                "source_train_sha256":parent["source_train_sha256"], "parent_seed":parent["seed"],
                "excluded_source_ids":parent["excluded_source_ids"], "batches":batches,
                "preceding_revision_manifest_sha256":sha256(revised / "final-manifest.json"),
                "usage":"Next planned train-pool wave; reviewed preannotation only, not test or human gold",
                "origin":"teacher_model_unreviewed", "human_gold":False, "training_ready":False,
                "files":{name:sha256(OUT / name) for name in files}}
    atomic_json(OUT / "manifest.json", manifest)
    report = {"records":100,"batches":100,"batch_size":1,"parent_batches":"0021-0040",
              "source_split":"clean-train", "prior100_record_and_group_overlap":0,
              "duplicate_groups":100,"unchanged_original_text_and_spans":True,"aspect_count":len(aspects),
              "length_strata":{"1-200":sum(n<=200 for n in lengths),"201-500":sum(200<n<=500 for n in lengths),
                               "501+":sum(n>500 for n in lengths)},
              "prepared_only":True,"model_calls":0,"human_gold":False,"training_ready":False,
              "remaining_after_this_wave":700,"manifest_sha256":sha256(OUT / "manifest.json")}
    atomic_json(ROOT / "docs/reports/stage3-wave100-v3.2-preparation.json", report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
