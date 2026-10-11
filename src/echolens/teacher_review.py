"""Validate externally generated annotations without granting human-gold status."""
from copy import deepcopy

from echolens.annotation import occurrences, validate_opinions


def normalize_web_response(payload):
    """Fill omitted empty coverage references only; retain an explicit repair log."""
    normalized, repairs = deepcopy(payload), []
    if not isinstance(normalized, dict) or not isinstance(normalized.get("records"), list):
        return normalized, repairs
    for record in normalized["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("coverage"), list):
            continue
        for entry in record["coverage"]:
            if isinstance(entry, dict) and entry.get("status") in ("context", "no_opinion", "uncertain") and "opinion_ids" not in entry:
                entry["opinion_ids"] = []
                repairs.append({"record_id": record.get("record_id"), "span_id": entry.get("span_id"),
                                "change": "filled omitted opinion_ids with []"})
    return normalized, repairs


def validate_response(payload, inputs):
    if not isinstance(payload, dict) or set(payload) != {"records"} or not isinstance(payload["records"], list):
        raise ValueError("输出必须是含 records 数组的 JSON 对象")
    sources = {r["record_id"]: r for r in inputs["records"]}
    records, seen, warnings = [], set(), []
    for result in payload["records"]:
        if not isinstance(result, dict):
            raise ValueError("每条结果必须是对象")
        rid = result.get("record_id")
        if rid not in sources or rid in seen:
            raise ValueError("结果包含未知或重复 record_id")
        seen.add(rid)
        source = sources[rid]
        text = source["text_raw"]
        if not isinstance(result.get("opinions"), list) or not isinstance(result.get("coverage"), list):
            raise ValueError("每条结果需要 opinions 和 coverage 数组")
        questions = result.get("review_questions")
        if not isinstance(questions, list) or any(not isinstance(q, str) for q in questions):
            raise ValueError("review_questions 必须为字符串数组")
        opinions = []
        opinion_ids = set()
        for raw in result["opinions"]:
            if not isinstance(raw, dict):
                raise ValueError("意见必须是对象")
            oid = raw.get("opinion_id")
            if not isinstance(oid, str) or not oid or oid in opinion_ids:
                raise ValueError("opinion_id 为空或重复")
            opinion_ids.add(oid)
            if any(not isinstance(raw.get(k), str) for k in ("target", "aspect", "sentiment", "intent", "scope", "evidence")):
                raise ValueError("意见的对象、标签及证据必须为字符串")
            matches = occurrences(text, raw["evidence"])
            nth = raw.get("occurrence")
            if type(nth) is not int or not 1 <= nth <= len(matches):
                raise ValueError("证据不在原文或 occurrence 无效")
            start = matches[nth - 1]
            op = {k: raw[k] for k in ("target", "aspect", "sentiment", "intent", "scope", "evidence")}
            op.update(opinion_id=oid, start=start, end=start + len(raw["evidence"]))
            opinions.append(op)
        validate_opinions(text, opinions, inputs["aspects"])
        by_op = {o["opinion_id"]: o for o in opinions}
        spans = {s["span_id"]: s for s in source["spans"]}
        covered, referenced = set(), set()
        for entry in result["coverage"]:
            if not isinstance(entry, dict):
                raise ValueError("coverage 条目必须为对象")
            sid, state, ids = entry.get("span_id"), entry.get("status"), entry.get("opinion_ids")
            if sid not in spans or sid in covered:
                raise ValueError("coverage 包含未知或重复片段")
            if state not in ("opinion", "context", "no_opinion", "uncertain") or not isinstance(ids, list):
                raise ValueError("coverage 状态或意见引用无效")
            if any(not isinstance(oid, str) or oid not in opinion_ids for oid in ids) or len(set(ids)) != len(ids):
                raise ValueError("coverage 引用了不存在或重复的意见")
            if state == "opinion" and not ids or state in ("no_opinion", "uncertain") and ids:
                raise ValueError("coverage 状态与引用不一致")
            if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
                raise ValueError("coverage 需要简短依据")
            if state == "opinion" and any(not (by_op[oid]["start"] < spans[sid]["end"] and by_op[oid]["end"] > spans[sid]["start"]) for oid in ids):
                raise ValueError("coverage 声称的意见证据不与该片段相交")
            if state == "opinion":
                referenced.update(ids)
            covered.add(sid)
        if covered != set(spans) or referenced != opinion_ids:
            raise ValueError("存在未核对片段或未关联任何片段的意见")
        for i, a in enumerate(opinions):
            for b in opinions[i+1:]:
                overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
                if overlap <= 0:
                    continue
                same = all(a[k] == b[k] for k in ("target", "aspect", "sentiment", "intent", "scope"))
                if same or (a["target"] != b["target"] and a["sentiment"] != b["sentiment"]):
                    warnings.append({"record_id": rid, "opinions": [a["opinion_id"], b["opinion_id"]],
                                     "reason": "相同意见证据重叠，检查重复" if same else "不同对象且情感不同的证据相交，检查边界"})
        records.append({"record_id": rid, "opinions": opinions, "coverage": result["coverage"], "review_questions": questions})
    if seen != set(sources):
        raise ValueError("输出缺少输入记录；请勿把截断结果当成完成")
    return {"origin": "teacher_model_unreviewed", "training_ready": False, "human_gold": False,
            "structural_checks_passed": True, "semantic_accuracy_established": False,
            "records": records, "warnings": warnings}
