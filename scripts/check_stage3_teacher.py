"""Check a saved Gemini JSON response and produce a local changes-only review."""
import argparse
import html
import json
from pathlib import Path

from echolens.annotation import digest
from echolens.experiments import ROOT, sha256
from echolens.teacher_review import normalize_web_response, validate_response
from echolens.workbench import atomic_json, read_json


def key(op):
    return tuple(op[k] for k in ("target", "aspect", "sentiment", "intent", "scope", "evidence"))


def show(op):
    title = " · ".join(op[k] for k in ("target", "aspect", "sentiment", "intent", "scope"))
    return f"<li><b>{html.escape(title)}</b><p>{html.escape(op['evidence'])}</p></li>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    parser.add_argument("--model", required=True, help="实际网页显示的模型名称，不推断后台版本")
    parser.add_argument("--thinking", default="扩展")
    parser.add_argument("--source-id", help="单条文件返回时填写来源ID；不填写则必须包含全部5条")
    args = parser.parse_args()
    folder = ROOT / "data/annotations/stage3-teacher-pilot-v1"
    manifest = read_json(folder / "manifest.json")
    for name, checksum in manifest["files"].items():
        if sha256(folder / name) != checksum:
            raise ValueError("审核输入或人工参考已改变，不能继续使用该试跑版本")
    inputs = read_json(folder / "inputs.json")
    if args.source_id:
        inputs["records"] = [r for r in inputs["records"] if r["source_record_id"] == args.source_id]
        if not inputs["records"]:
            raise ValueError("来源ID不在本次5条试跑包内")
    raw = args.response.read_text(encoding="utf-8-sig").strip()
    if raw.startswith("```json\n") and raw.endswith("```"):
        raw = raw[len("```json\n"):-3].strip()
    parsed = json.loads(raw)
    normalized, repairs = normalize_web_response(parsed)
    checked = validate_response(normalized, inputs)
    checked["format_repairs"] = repairs
    checked["raw_schema_passed"] = not repairs
    run_info = {"response_sha256": sha256(args.response), "input_sha256": digest(inputs),
                "model_user_reported": args.model, "thinking_user_reported": args.thinking,
                "provider": "Gemini web", "billing_usage_available": False,
                "normalizer_version": "empty-coverage-refs-v1"}
    checked["provenance"] = run_info
    output = folder / "responses" / digest(run_info)[:16]
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw-response.txt").write_bytes(args.response.read_bytes())
    atomic_json(output / "normalized-response.json", normalized)
    atomic_json(output / "validated-response.json", checked)
    references = read_json(folder / "human-reference-DO-NOT-UPLOAD.json")
    sources = {r["record_id"]: r for r in inputs["records"]}
    parts = ["<html lang='zh'><meta charset='utf-8'><title>AI复核差异</title><style>body{font-family:sans-serif;max-width:1200px;margin:30px auto;line-height:1.7}section{padding:20px;border:1px solid #ddd;margin:20px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}p{white-space:pre-wrap}li{margin-bottom:14px}</style><h1>AI复核差异 · 尚未人工确认</h1><p>精确字段比较，文字改写也可能显示为差异。不是准确率；旧人工试标也可能漏标。本页不修改人工记录。请重点核验新增、移除/修改和复核问题，同时抽查没有变化的内容。</p>"]
    for record in checked["records"]:
        rid = record["record_id"]
        before = (references.get(rid) or {}).get("opinions", [])
        after = record["opinions"]
        old_keys, new_keys = {key(o) for o in before}, {key(o) for o in after}
        added, removed = [o for o in after if key(o) not in old_keys], [o for o in before if key(o) not in new_keys]
        parts.append(f"<section><h2>记录 {html.escape(sources[rid]['source_record_id'])}</h2><details><summary>完整原文</summary><p>{html.escape(sources[rid]['text_raw'])}</p></details><div class='cols'><div><h3>原人工试标中被移除或修改</h3><ul>{''.join(show(o) for o in removed)}</ul></div><div><h3>AI新增或修改（未确认）</h3><ul>{''.join(show(o) for o in added)}</ul></div></div><p>完全一致意见：{len(old_keys & new_keys)} 条</p><h3>模型复核问题</h3><pre>{html.escape(json.dumps(record['review_questions'],ensure_ascii=False,indent=2))}</pre></section>")
    parts.append("<h2>本地重叠提示（非自动判错）</h2><pre>" + html.escape(json.dumps(checked["warnings"], ensure_ascii=False, indent=2)) + "</pre></html>")
    (output / "review-diff.html").write_text("".join(parts), encoding="utf-8")
    print(f"Validated {len(checked['records'])} records; empty-array repairs={len(repairs)}; overlap warnings={len(checked['warnings'])}; semantic accuracy NOT established")
    print(output / "review-diff.html")


if __name__ == "__main__":
    main()
