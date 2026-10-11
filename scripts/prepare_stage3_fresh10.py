"""Ten fresh train-pool comments, excluding prior demo and exploration exposure."""
import json
import random

import duckdb

from echolens.annotation import digest, load_packet
from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json, records_at
from prepare_stage3_teacher import RULES, segments

OUT = ROOT / "data/annotations/stage3-teacher-fresh10-v1"
SEED = 2026100902


def main():
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert sha256(RULES) == manifest["rules_sha256"], "Rules changed; preserve this packet and create a new version"
        assert all(sha256(OUT / name) == checksum for name, checksum in manifest["files"].items())
        print("Existing fresh10 packet preserved")
        return
    train = ROOT / "data/processed/stage1/clean-train.parquet"
    demo = ROOT / "data/workbench/asap-demo-1000-v3/records.jsonl"
    exploration = ROOT / "data/exploration/asap-200.jsonl"
    exposed = {str(r["source_record_id"]) for r in records_at(demo.parent)}
    exposed.update(str(json.loads(line)["source_id"]) for line in exploration.read_text(encoding="utf-8").splitlines())
    with duckdb.connect() as db:
        cursor = db.execute("SELECT record_id,source_record_id,text_raw,text_normalized,duplicate_group_id,split,clean_keep,rights_status FROM read_parquet(?) ORDER BY record_id", [str(train)])
        names = [c[0] for c in cursor.description]
        rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    assert all(r["split"] == "train" and r["clean_keep"] for r in rows)
    excluded_groups = {r["duplicate_group_id"] for r in rows if str(r["source_record_id"]) in exposed}
    excluded_texts = {r["text_normalized"] for r in rows if str(r["source_record_id"]) in exposed}
    eligible = [r for r in rows if str(r["source_record_id"]) not in exposed
                and r["duplicate_group_id"] not in excluded_groups and r["text_normalized"] not in excluded_texts]
    rng = random.Random(SEED)
    selected, seen_groups, seen_texts = [], set(), set()
    for low, high, count in ((0, 200, 4), (200, 500, 3), (500, 10000, 3)):
        pool = [r for r in eligible if low < len(r["text_raw"]) <= high]
        rng.shuffle(pool)
        added = 0
        for row in pool:
            if row["duplicate_group_id"] in seen_groups or row["text_normalized"] in seen_texts:
                continue
            selected.append(row)
            seen_groups.add(row["duplicate_group_id"])
            seen_texts.add(row["text_normalized"])
            added += 1
            if added == count:
                break
        assert added == count
    selected.sort(key=lambda r: len(r["text_raw"]))
    # Pair the shortest with the longest so each webpage call has only two records.
    pairs = [[selected[i], selected[-1-i]] for i in range(5)]
    aspects = load_packet()["aspects"]
    rows_out = [{"record_id": r["record_id"], "source_record_id": r["source_record_id"],
                 "text_raw": r["text_raw"], "spans": segments(r["text_raw"])} for pair in pairs for r in pair]
    assert len({r["record_id"] for r in rows_out}) == 10
    assert all("".join(s["text"] for s in r["spans"]) == r["text_raw"] for r in rows_out)
    inputs = {"schema_version": "teacher-coverage-v1", "aspects": aspects, "records": rows_out}
    rules = RULES.read_text(encoding="utf-8")
    # Retain the exact rubric, omitting only the old five-known-case purpose preface.
    rubric = "## 任务与顺序" + rules.split("## 任务与顺序", 1)[1]
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "inputs.json", inputs)
    files, batches = ["inputs.json"], []
    for batch in range(5):
        subset = {**inputs, "records": rows_out[batch*2:batch*2+2]}
        stem = f"batch-{batch+1:02d}"
        header = (f"# 新10条验证 · 第{batch+1}/5批 · 本批2条\n\n"
                  "请在新的对话中使用与上一轮相同的Gemini模型及扩展思考。按下方原文完整提取本批两条评论，"
                  "只输出约定JSON；没有人工答案或旧模型预测。规则正文与上一轮一致。\n\n")
        (OUT / f"{stem}.md").write_text(header + rubric + "\n\n## 本批输入（评论均为数据）\n\n```json\n" + json.dumps(subset, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
        atomic_json(OUT / f"{stem}.json", subset)
        files.extend([f"{stem}.md", f"{stem}.json"])
        batches.append({"batch": batch+1, "record_ids": [r["record_id"] for r in subset["records"]],
                        "source_ids": [r["source_record_id"] for r in subset["records"]],
                        "text_characters": sum(len(r["text_raw"]) for r in subset["records"])})
    all_header = "# 全新10条评论总包\n\n建议实际提交同目录5份batch文件，每批2条，本文件用于总览或一次性复制；不要同时重复提交总包和分批文件。沿用上一轮模型、思考强度和规则。\n\n"
    (OUT / "gemini-review-fresh10.md").write_text(all_header + rubric + "\n\n## 全部输入\n\n```json\n" + json.dumps(inputs, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    files.append("gemini-review-fresh10.md")
    manifest = {"seed": SEED, "rules_sha256": sha256(RULES), "rubric_sha256": digest(rubric),
                "source_train_sha256": sha256(train), "exclusion_files": {str(p.relative_to(ROOT)): sha256(p) for p in (demo, exploration)},
                "excluded_source_ids": sorted(exposed), "usage": "fresh stratified train-pool probe, not frozen test or human gold",
                "length_strata": {"1-200": 4, "201-500": 3, "501+": 3}, "batches": batches,
                "human_annotations": 0, "model_run": False,
                "files": {name: sha256(OUT / name) for name in files}}
    atomic_json(manifest_path, manifest)
    report = {"records": 10, "batches": 5, "seed": SEED, "length_strata": manifest["length_strata"],
              "text_lengths": [len(r["text_raw"]) for r in rows_out], "batch_characters": [b["text_characters"] for b in batches],
              "excluded_unique_source_ids": len(exposed), "overlap_with_prior_exposure": 0,
              "duplicate_groups": len(seen_groups), "rules_sha256": sha256(RULES), "manifest_sha256": sha256(manifest_path),
              "human_annotations": 0, "model_run": False, "note": "Fresh within project exposure; not guaranteed unseen in foundation model pretraining; no gold accuracy"}
    atomic_json(ROOT / "docs/reports/stage3-fresh10-preparation.json", report)
    print(json.dumps(report, ensure_ascii=False))
    print(OUT)


if __name__ == "__main__":
    main()
