"""Render local AI notes and publish only aggregate exploration statistics."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/exploration/asap-200-ai-notes.jsonl"
notes = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
samples = {r["source_id"]: r for r in (json.loads(line) for line in (ROOT / "data/exploration/asap-200.jsonl").read_text(encoding="utf-8").splitlines())}
assert len(notes) == len({r["source_id"] for r in notes}) == 200
assert set(samples) == {r["source_id"] for r in notes}
valid = sum(r["valid"] for r in notes)
parsed = sum("note" in r and isinstance(r["note"], dict) for r in notes)
seconds = sorted(r["seconds"] for r in notes)
report = {"rows": len(notes), "model_revision": notes[0]["model_revision"], "prompt_version": "reading-v1", "json_objects_parsed": parsed, "structure_and_verbatim_passed": valid, "structure_and_verbatim_pass_rate": valid / len(notes), "needs_review_ids": [r["source_id"] for r in notes if not r["valid"]], "generation_seconds": sum(seconds), "p50_seconds": statistics.median(seconds), "p95_seconds": seconds[int(.95 * (len(seconds) - 1))], "human_gold_records": 0, "limitations": ["Train-only star-stratified exploration", "Lexical checks do not evaluate semantic correctness", "No manual gold labels; no held-out quality metric"]}
(ROOT / "docs/reports/exploration-reading.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
lines = ["# 200 条探索样本：AI 初读，待人工复核", "", "仅训练集。不是金标准；引用通过也不表示概述正确。本文件含评论片段，仅供本地审核，不提交 Git。", ""]
for item in notes:
    lines += [f'## {item["sample_index"]}. 记录 {item["source_id"]}', "", f'自动结构/原文匹配：{"通过" if item["valid"] else "失败，需复核"}', ""]
    if "note" in item and isinstance(item["note"], dict):
        for key, title in [("note", "AI 概述"), ("evidence", "模型引用"), ("caution", "模型提示")]:
            lines.append(f'{title}：{item["note"].get(key, "缺失")}')
            lines.append("")
    else:
        lines += [item["output"], ""]
    lines += ["<details><summary>原始评论</summary>", "", samples[item["source_id"]]["text"], "", "</details>", ""]
target = ROOT / "artifacts/stage-00/exploration-200.md"
target.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))

# Optional paired prompt experiment, always on the same training records.
repair_path = ROOT / "data/exploration/asap-ai-notes-v2.jsonl"
if repair_path.exists():
    repaired = [json.loads(line) for line in repair_path.read_text(encoding="utf-8").splitlines()]
    baseline = {r["source_id"]: r for r in notes}
    assert repaired and len(repaired) == len({r["source_id"] for r in repaired})
    assert all(r["source_id"] in baseline and r["prompt_version"] == "reading-v2" and r["model_revision"] == baseline[r["source_id"]]["model_revision"] for r in repaired)
    comparison = {
        "rows": len(repaired),
        "model_revision": repaired[0]["model_revision"],
        "source_ids": [r["source_id"] for r in repaired],
        "baseline_prompt": "reading-v1",
        "revised_prompt": "reading-v2",
        "baseline_structure_and_verbatim_passed": sum(baseline[r["source_id"]]["valid"] for r in repaired),
        "revised_structure_and_verbatim_passed": sum(r["valid"] for r in repaired),
        "revised_failed_ids": [r["source_id"] for r in repaired if not r["valid"]],
        "limitations": [
            "Same training exploration records used for prompt repair; not independent evaluation",
            "Revised prompt requests only one opinion, reducing coverage",
            "Checks require exact keys, non-empty strings and verbatim evidence; do not enforce requested character limits or semantic support",
            "No human gold labels or fine-tuning quality claim",
        ],
    }
    (ROOT / "docs/reports/evidence-prompt-comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
