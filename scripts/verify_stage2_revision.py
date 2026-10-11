"""Verify v2 provenance and constraints; export a private, non-gold comparison review."""
import argparse
import html
import json
from collections import Counter
from echolens.experiments import ROOT, sha256, save_json
from echolens.workbench import (dataset_path, records_at, read_json, results_at, highlighted,
                               topic_state, report_markdown, OPINION_LABELS, SENTIMENTS)
from echolens.topic_groups import representative_units


def summarize(results, ids):
    selected = [results[rid] for rid in ids if rid in results]
    valid = [r for r in selected if r["valid"]]
    opinions = [op for r in valid for op in r["opinions"]]
    return {"attempted":len(selected),"records_with_accepted_extraction":len(valid),"opinions":len(opinions),
            "records_with_opinions":sum(bool(r["opinions"]) for r in valid),
            "valid_empty_records":sum(not r["opinions"] for r in valid),
            "fully_valid_records":sum(r.get("fully_valid",r["valid"]) for r in selected),
            "partial_records":sum(r.get("partial",False) for r in selected),
            "rejected_opinion_errors":dict(Counter(op["error"] for r in selected for op in r.get("rejected",[]))),
            "failures":dict(Counter(r["error"] for r in selected if not r["valid"])),
            "target_present":sum(bool(o.get("target")) for o in opinions),
            "grounding_methods":dict(Counter(o.get("grounding_method","sentence_id") for o in opinions)),
            "scope_uncertain":sum(o.get("scope")=="uncertain" for o in opinions),
            "merged_opinions":sum(len(o.get("raw_indices",[])) > 1 for o in opinions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    old, new = dataset_path("asap-demo-1000"), dataset_path("asap-demo-1000-v2")
    rows = records_at(new)
    by_id = {r["record_id"]:r for r in rows}
    baseline, revised = results_at(old), results_at(new)
    reuse = read_json(new/"reused-windows.json", {})
    if reuse:
        current_run = read_json(new/"analysis-run.json")
        for origin in (reuse["source_run"], reuse["pilot_source_run"]):
            for key in ("snapshot_sha256", "models", "classifier_sha256", "prompt_sha256"):
                assert origin[key] == current_run[key]
            for key in ("window_chars", "context_clauses", "max_new_tokens", "quantized", "enable_thinking"):
                assert origin["config"][key] == current_run["config"][key]
        for name, digest in reuse["windows"].items():
            assert sha256(new/"windows"/name) == digest
    assert sha256(old/"records.jsonl") == sha256(new/"records.jsonl")
    assert sha256(old/"classifications.json") == sha256(new/"classifications.json")
    for name, digest in read_json(new/"baseline-preservation.json").items():
        assert sha256(old/name) == digest
    if not args.partial:
        assert len(revised) == len(rows) == 1000
    ids = sorted(revised)
    for rid, result in revised.items():
        if result["valid"]:
            highlighted(by_id[rid]["text_raw"],result["opinions"])
            for op in result["opinions"]:
                assert op["target"] in op["evidence"]
                assert op["aspect"] in OPINION_LABELS
    state = topic_state(new)
    if not args.partial:
        expected = {f"{rid}:{i}" for rid,r in revised.items() if r["valid"] for i in range(len(r["opinions"]))}
        assert expected == set(state["assignments"]) == {u["unit_id"] for u in state["units"]}
    for topic in state["topics"]:
        units = [u for u in state["units"] if state["assignments"][u["unit_id"]] == topic["id"]]
        assert len({u["aspect"] for u in units}) == 1
        assert all(u.get("scope","current")=="current" for u in units)
        assert len({u["record_id"] for u in units}) >= 3
        assert topic["max_pairwise_cosine_distance"] <= .30001
        reps = representative_units(units)
        assert len({u["record_id"] for u in reps}) == len(reps)
    selection = read_json(new/"audit-selection.json")
    case_ops = {r["source_record_id"]:revised.get(r["record_id"],{}).get("opinions",[]) for r in rows if r["record_id"] in selection["user_reported"]}
    case_checks = {
        "38689_decoration_negative":any(o["aspect"]=="Ambience#Decoration" and o["sentiment"]==-1 and "装修" in o["evidence"] for o in case_ops["38689"]),
        "6479_table_space_negative":any(o["aspect"]=="Ambience#Space" and o["sentiment"]==-1 and "桌子" in o["evidence"] for o in case_ops["6479"]),
        "44572_parking_negative":any(o["aspect"]=="Service#Parking" and o["sentiment"]==-1 for o in case_ops["44572"]),
        "32524_parking_negative":any(o["aspect"]=="Service#Parking" and o["sentiment"]==-1 for o in case_ops["32524"]),
        "5365_linked_praise_has_specific_dish":sum(o["target"]=="石锅海鲜饭" and o["sentiment"]==1 and "重点推荐" in o["evidence"] and "果然没错" in o["evidence"] for o in case_ops["5365"])==1,
        "5365_pineapple_ingredients_not_portion":any(o["aspect"]=="Food#Ingredients" and o["sentiment"]==-1 and o["target"]=="菠萝海鲜饭" for o in case_ops["5365"]) and not any(o["aspect"]=="Food#Portion" and "卷心菜" in o["evidence"] for o in case_ops["5365"]),
    }
    assert all(case_checks.values()), "User-reported development regressions need review"
    report = {"scope":"train-only development comparison; not held-out or human-gold accuracy",
              "status":"partial" if args.partial else "complete",
              "same_input_snapshot":True,"unchanged_classifier":True,"baseline_files_unchanged":True,
              "reused_raw_windows":len(reuse.get("windows",{})),
              "user_reported_development_case_checks":case_checks,
              "total_windows":len(list((new/"windows").glob("*.json"))),
              "v1_on_same_records":summarize(baseline,ids),"v2":summarize(revised,ids),
              "pilot":summarize(revised,selection["pilot"]),"additional_audit":summarize(revised,selection["additional_audit"]),
              "topics":len(state["topics"]),"topic_sizes":dict(Counter(state["assignments"].values())),
              "topic_aspects":dict(Counter(t.get("aspect","unknown") for t in state["topics"])),
              "topic_pipeline_seconds":state.get("seconds"),
              "no_cross_predicted_aspect_topics":True,"distinct_record_representatives":True,
              "elapsed_generation_batch_seconds":sum(r["seconds"] for r in revised.values()),
              "limitations":["Aspect partitions reflect predictions, not proof of semantic correctness", "Targets and quoted spans are checked for existence, not entailment", "Topic names and opinion attribution need human review", "New food labels do not modify the trained 18-aspect classifier", "Pilot and audit use train data; no gold accuracy claimed", "Retained valid items do not imply that raw model output passed the full schema"]}
    suffix = "pilot" if args.partial else "workflow"
    save_json(ROOT/f"docs/reports/stage2-v2-{suffix}.json", report)
    pieces = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>阶段2修订对照</title><style>body{font:16px/1.8 system-ui;max-width:1200px;margin:40px auto;padding:20px;background:#f7f8fa;color:#25363f}article{background:white;padding:20px;margin:16px 0;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{display:grid;grid-template-columns:1fr 1fr;gap:24px}blockquote{margin:8px 0 24px;padding:12px;background:#edf3f5;white-space:pre-wrap}small{color:#526770}nav a{margin-right:12px}@media(max-width:700px){section{grid-template-columns:1fr}}</style><h1>阶段 2 修订对照 · 待人工审核</h1><p>含用户指出案例、开发探针及另一批固定随机训练样本；不是人工金标准，不用于独立测试成绩。请核验方面、对象和证据支持关系；引用存在不等于语义正确。</p>']
    audit_ids = list(dict.fromkeys(selection["pilot"]+selection["additional_audit"]))
    pieces.append('<nav>'+''.join(f'<a href="#r-{html.escape(str(by_id[rid]["source_record_id"]))}">{html.escape(str(by_id[rid]["source_record_id"]))}</a>' for rid in audit_ids)+'</nav>')
    for rid in audit_ids:
        row = by_id[rid]
        pieces.append(f'<article id="r-{html.escape(str(row["source_record_id"]))}"><h2>记录 {html.escape(str(row["source_record_id"]))}</h2><details><summary>完整原文</summary><pre>{html.escape(row["text_raw"])}</pre></details><section>')
        for name, result in (("v1",baseline.get(rid)),("v2",revised.get(rid))):
            view = {key:result.get(key) for key in ("valid","fully_valid","partial","error","review_flags","rejected","opinions")} if result else "未运行"
            status = "未运行" if not result else "部分意见待复核" if result.get("partial") else "字段与定位检查通过，语义待审" if result["valid"] else "抽取失败"
            pieces.append(f'<div><h3>{name}</h3><p>{status}</p>')
            for op in result.get("opinions",[]) if result else []:
                target = op.get("target") or "未明确对象"
                scope = {"current":"当前餐厅（机器判断）","other":"其他对象","uncertain":"归属待核验"}.get(op.get("scope"),"旧版未区分归属")
                pieces.append(f'<strong>{OPINION_LABELS[op["aspect"]]} · {SENTIMENTS[op["sentiment"]]}</strong><br><small>对象：{html.escape(target)} · {scope}</small><blockquote>{html.escape(op["evidence"])}</blockquote>')
            pieces.append(f'<details><summary>校验原因与完整字段</summary><pre>{html.escape(json.dumps(view,ensure_ascii=False,indent=2))}</pre></details></div>')
        pieces.append('</section></article>')
    pieces.append('</html>')
    (ROOT/"artifacts/stage-02/revision-comparison.html").write_text(''.join(pieces),encoding="utf-8",newline="\n")
    if not args.partial:
        (ROOT/"artifacts/stage-02/feedback-report-v2.md").write_text(report_markdown(new,rows,read_json(new/"classifications.json"),revised),encoding="utf-8",newline="\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
