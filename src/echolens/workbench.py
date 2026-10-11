"""Local workbench storage, evidence validation and append-only review history."""
import hashlib
import html
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import psutil

from echolens.experiments import ROOT, save_json
from echolens.ingest import read_records, import_records, quality_summary

STORE = ROOT / "data/workbench"
TAXONOMY = json.loads((ROOT / "configs/labels/asap-v0.1.json").read_text(encoding="utf-8"))
ASPECTS = list(TAXONOMY["aspects"])
SENTIMENTS = {-2: "未提及", -1: "负面", 0: "中性", 1: "正面"}
OPINION_LABELS = {**TAXONOMY["aspects"], "Food#Ingredients": "食材选择与搭配", "Food#Preparation": "制作方式", "unknown": "未覆盖 / 待定"}
LEGACY_OPINION_LABELS = dict(OPINION_LABELS)
OPINION_LABELS.update(json.loads((ROOT/'configs/labels/teacher-v3.json').read_text(encoding='utf-8'))['aspects'])
SENTIMENTS.update({2:'混合',3:'不确定'})


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    save_json(temp, value)
    temp.replace(path)


def dataset_path(dataset_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", dataset_id):
        raise ValueError("Invalid dataset ID")
    return STORE / dataset_id


def datasets():
    return [read_json(p) for p in sorted(STORE.glob("*/dataset.json"))]


def job_running(folder):
    job = read_json(Path(folder) / "job.json", {})
    if job.get("status") == "running" and job.get("pid"):
        return psutil.pid_exists(job["pid"])
    progress = read_json(Path(folder) / "progress.json", {})
    if progress.get("status") != "running":
        return False
    return psutil.pid_exists(progress["pid"]) if progress.get("pid") else True


def records_at(folder):
    return [json.loads(line) for line in (Path(folder) / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def save_dataset(folder, records, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(folder / "feedback.duckdb")) as connection:
        inserted = import_records(connection, records)
        assert import_records(connection, records) == 0
        connection.sql("SELECT * FROM feedback ORDER BY record_id").write_parquet(str(folder / "feedback.parquet"))
    records.sort(key=lambda r: r["record_id"])
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    (folder / "records.jsonl").write_text(text, encoding="utf-8", newline="\n")
    metadata.update({"dataset_id": folder.name, "rows": len(records), "snapshot_sha256": hashlib.sha256(text.encode()).hexdigest(), "quality": quality_summary(records)})
    atomic_json(folder / "dataset.json", metadata)
    return inserted


def import_upload(path, mapping, source, revision, rights):
    rows, stats = read_records(path, mapping=mapping, source=source, revision=revision, rights_status=rights)
    if not rows:
        raise ValueError("没有可导入的非空反馈，请检查字段映射。")
    identity = json.dumps([source, revision, mapping], sort_keys=True, ensure_ascii=False)
    dataset_id = "upload-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    folder = dataset_path(dataset_id)
    existing = {r["record_id"]: r for r in records_at(folder)} if (folder / "records.jsonl").exists() else {}
    previous_count = len(existing)
    for row in rows:
        previous = existing.get(row["record_id"])
        if previous and any(previous[k] != row[k] for k in row if k != "source_file_sha256"):
            raise ValueError("同一来源版本中的记录内容改变，请使用新版本。")
        existing.setdefault(row["record_id"], row)
    if (folder / "analysis-run.json").exists() and len(existing) != read_json(folder / "dataset.json")["rows"]:
        raise ValueError("数据已分析，请使用新版本导入新增记录，以保留原有分析和审核。")
    stats["new_records"] = len(existing)-previous_count
    metadata = {"name": f"{source} · {revision}", "kind": "uploaded", "import_stats": stats}
    previous_metadata = read_json(folder / "dataset.json", {})
    metadata["analysis_config"] = previous_metadata.get("analysis_config", "configs/stage2.json" if existing and (folder / "analysis-run.json").exists() else "configs/stage2-v2.json")
    save_dataset(folder, list(existing.values()), metadata)
    return dataset_id, stats


def sentences(text):
    # Keep exact positions, including literal escaped newlines; never normalize evidence.
    spans = []
    for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;\n]*|[。！？!?；;\n]+", text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end-1].isspace():
            end -= 1
        if start < end:
            spans.append({"id": len(spans), "start": start, "end": end, "text": text[start:end]})
    return spans


def decode_opinions(output, text, max_opinions=3):
    payload = json.loads(output)
    if not isinstance(payload, dict) or set(payload) != {"opinions"} or not isinstance(payload["opinions"], list):
        raise ValueError("输出必须是仅包含 opinions 数组的 JSON")
    if len(payload["opinions"]) > max_opinions:
        raise ValueError("意见数量超过上限")
    spans, opinions, seen = sentences(text), [], set()
    for item in payload["opinions"]:
        if not isinstance(item, dict) or set(item) != {"a", "s", "i", "e"}:
            raise ValueError("意见字段无效")
        a, s, e = item["a"], item["s"], item["e"]
        if type(a) is not int or not 0 <= a < len(ASPECTS) or type(s) is not int or s not in (-1, 0, 1):
            raise ValueError("方面或情感标签无效")
        if item["i"] not in TAXONOMY["intents"] or type(e) is not int or not 0 <= e < len(spans):
            raise ValueError("意图或原文片段编号无效")
        span = spans[e]
        key = (a, s, item["i"], e)
        if key in seen:
            raise ValueError("重复意见")
        seen.add(key)
        opinions.append({"aspect": ASPECTS[a], "sentiment": s, "intent": item["i"],
                         "evidence": span["text"], "start": span["start"], "end": span["end"]})
    return opinions


def highlighted(text, opinions):
    spans = []
    for op in sorted(opinions, key=lambda x: x["start"]):
        start, end = op["start"], op["end"]
        if not (0 <= start < end <= len(text)) or text[start:end] != op["evidence"]:
            raise ValueError("证据与原文位置不一致")
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(end, spans[-1][1])
        else:
            spans.append([start, end])
    cursor, parts = 0, []
    for start, end in spans:
        parts.extend([html.escape(text[cursor:start]), "<mark>", html.escape(text[start:end]), "</mark>"])
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return '<div class="evidence-text">' + "".join(parts).replace("\n", "<br>") + "</div>"


def results_at(folder):
    return {p.stem: read_json(p) for p in (Path(folder) / "extractions").glob("*.json")}


def review_reasons(record_id, classifications, extractions, threshold=.65):
    reasons = []
    prediction = classifications.get(record_id)
    extracted = extractions.get(record_id)
    if extracted is None:
        reasons.append("待抽取")
    elif not extracted["valid"]:
        reasons.append("抽取失败")
    elif extracted.get("partial"):
        reasons.append("部分意见失败")
    if extracted and extracted.get('empty_result'):
        reasons.append('空结果需检查遗漏')
    if prediction:
        if any(p < threshold for v, p in zip(prediction["labels"], prediction["confidence"]) if v != -2):
            reasons.append("低把握")
        if extracted and extracted["valid"] and any(o["aspect"] in ASPECTS and prediction["labels"][ASPECTS.index(o["aspect"])] != o["sentiment"] for o in extracted["opinions"]):
            reasons.append("模型分歧")
    if extracted and extracted["valid"] and any(o["aspect"] == "unknown" or ("target" in o and not o["target"]) for o in extracted["opinions"]):
        reasons.append("方面或对象待定")
    if int(record_id[:8], 16) % 20 == 0:
        reasons.append("随机抽查")
    return reasons


def events(folder):
    return [read_json(p) for p in sorted((Path(folder) / "events").glob("*.json"))]


def add_event(folder, kind, payload, actor):
    if not actor.strip():
        raise ValueError("请填写审核者")
    event = {"event_id": uuid.uuid4().hex, "at": datetime.now(timezone.utc).isoformat(), "kind": kind, "actor": actor.strip(), "payload": payload}
    (Path(folder) / "events").mkdir(exist_ok=True)
    atomic_json(Path(folder) / "events" / f"{event['at'].replace(':','-')}-{event['event_id']}.json", event)
    return event


def latest_reviews(folder):
    result = {}
    for event in events(folder):
        if event["kind"] == "review":
            result[event["payload"]["record_id"]] = event
    return result


def save_review(folder, record_id, opinions, decision, note, actor):
    record = next(r for r in records_at(folder) if r["record_id"] == record_id)
    candidate = read_json(Path(folder)/'dataset.json').get('analysis_config')=='configs/stage3-candidate.json'
    if decision not in ("approved", "edited", "rejected"):
        raise ValueError("Invalid review decision")
    for op in opinions:
        if op["aspect"] not in (OPINION_LABELS if candidate else LEGACY_OPINION_LABELS) or op["sentiment"] not in ((-1,0,1,2,3) if candidate else (-1,0,1)) or op["intent"] not in TAXONOMY["intents"]:
            raise ValueError("审核标签无效")
        if "target" in op and (not isinstance(op["target"],str) or (not candidate and op["target"] not in op["evidence"]) or (candidate and not op['target'].strip())):
            raise ValueError("审核评价对象不在所选证据中")
        if "issue" in op and (not isinstance(op["issue"],str) or not op["issue"].strip()):
            raise ValueError("具体意见不能为空")
        if op.get("scope", "current") not in (("current","other","past","uncertain") if candidate else ("current", "other", "uncertain")):
            raise ValueError("审核归属无效")
    highlighted(record["text_raw"], opinions)
    kind = read_json(Path(folder) / "dataset.json")["kind"]
    return add_event(folder, "review", {"record_id": record_id, "decision": decision, "opinions": opinions,
                     "note": note, "origin": "automated_fixture" if kind == "synthetic_fixture" else "human_review"}, actor)


def topic_state(folder):
    base = read_json(Path(folder) / "topics.json", {"version": 0, "topics": [], "units": [], "assignments": {}})
    for event in events(folder):
        if event["kind"] == "topics":
            base = event["payload"]
    return base


def edit_topics(folder, operation, ids, name, unit_ids, actor):
    state = topic_state(folder)
    topics = {str(t["id"]): t for t in state["topics"]}
    if not ids or any(str(i) not in topics for i in ids):
        raise ValueError("请先选择有效主题")
    if not name.strip():
        raise ValueError("主题名称不能为空")
    if operation == "rename":
        topics[str(ids[0])]["name"] = name.strip()
    elif operation == "merge":
        if len(set(ids)) < 2:
            raise ValueError("合并至少需要两个主题")
        target = str(ids[0])
        for uid, tid in state["assignments"].items():
            if str(tid) in {str(i) for i in ids}:
                state["assignments"][uid] = target
        for tid in ids[1:]:
            topics.pop(str(tid))
        topics[target]["name"] = name.strip()
    elif operation == "split":
        source = str(ids[0])
        available = {uid for uid, tid in state["assignments"].items() if str(tid) == source}
        if not unit_ids or not set(unit_ids) < available:
            raise ValueError("拆分应选择主题中的部分意见，至少留下一条")
        target = "manual-" + uuid.uuid4().hex[:8]
        topics[target] = {"id": target, "name": name.strip(), "keywords": [], "origin": "manual"}
        for uid in unit_ids:
            state["assignments"][uid] = target
    else:
        raise ValueError("Invalid topic edit")
    state["topics"] = list(topics.values())
    state["version"] += 1
    return add_event(folder, "topics", state, actor)


def aspect_counts(records, classifications):
    counts = []
    for i, aspect in enumerate(ASPECTS):
        labels = [classifications[r["record_id"]]["labels"][i] for r in records if r["record_id"] in classifications]
        values = Counter(labels)
        counts.append({"方面": TAXONOMY["aspects"][aspect], "aspect": aspect, "已分类分母": len(labels),
                       "提及": len(labels)-values[-2], "负面": values[-1], "中性": values[0], "正面": values[1]})
    return counts


def opinion_counts(records, extractions):
    """Distinct feedback counts; invalid, past and other-object opinions are excluded."""
    ids = {r['record_id'] for r in records}
    grouped = {}
    for rid,result in extractions.items():
        if rid not in ids or not result['valid']:
            continue
        for op in result['opinions']:
            if op.get('scope','current')!='current' or op['aspect']=='unknown' or op['sentiment']==3:
                continue
            key = op['aspect']
            bucket = grouped.setdefault(key,{'提及':set(),**{name:set() for name in ['负面','中性','正面','混合']}})
            bucket['提及'].add(rid)
            bucket[SENTIMENTS[op['sentiment']]].add(rid)
    return [{'方面':OPINION_LABELS[a],'aspect':a,'已处理分母':len(ids & extractions.keys()),
             **{k:len(v) for k,v in counts.items()}} for a,counts in sorted(grouped.items())]


def report_markdown(folder, records, classifications, extractions):
    metadata = read_json(Path(folder) / "dataset.json")
    candidate = metadata.get('analysis_config')=='configs/stage3-candidate.json'
    ids = {r["record_id"] for r in records}
    selected = {k: v for k, v in extractions.items() if k in ids}
    valid = {k: v for k, v in selected.items() if v["valid"]}
    reviews = latest_reviews(folder)
    lines = [f"# {metadata['name']} · 反馈分析", "", f"范围：筛选后的 {len(records)} 条反馈；已分类 {sum(r['record_id'] in classifications for r in records)} 条；抽取完成 {len(selected)} 条，其中有可用抽取 {len(valid)} 条。", "",
             f"全部通过字段与定位检查 {sum(v.get('fully_valid',v['valid']) for v in selected.values())} 条；部分意见待复核 {sum(v.get('partial',False) for v in selected.values())} 条。不是语义准确率。", "",
             "统计单位是反馈记录，不是用户数。分类统计使用固定 18 方面编码器预测；抽取为显著意见子集，不代表完整召回。引用存在已校验，语义及归因仍需人工复核。", "",
             "部分模型引用经过唯一空白/省略片段对齐后恢复为连续原文；这不证明意见语义正确。", "",
             f"数据快照：{metadata['snapshot_sha256']}；主题版本：{topic_state(folder)['version']}。", "",
             "没有提供真实时间字段时不生成趋势。审核历史单独导出，不覆盖机器统计；下方证据优先采用最新人工修正，拒绝项不输出证据。", "",
             "| 方面 | 已分类分母 | 提及 | 负面 | 中性 | 正面 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in aspect_counts(records, classifications):
        lines.append("| " + " | ".join(str(c[k]) for k in ("方面", "已分类分母", "提及", "负面", "中性", "正面")) + " |")
    if candidate:
        lines = [f"# {metadata['name']} · 反馈分析",'',f'范围：{len(records)}条；已抽取{len(selected)}条；通过机械检查{len(valid)}条；待复核失败{len(selected)-len(valid)}条。',
            '', '以下为QLoRA候选意见统计，只计当前对象/本次体验、通过字段与逐字引用检查的结果；不是语义准确率或用户数。同一反馈可有多个情感，情感列合计可能超过提及数。',
            '',f"数据快照：{metadata['snapshot_sha256']}；模型来源：candidate-run.json；无真实日期不生成趋势。",
            '', '| 方面 | 已处理分母 | 提及 | 负面 | 中性 | 正面 | 混合 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for c in opinion_counts(records,extractions):
            lines.append('| '+' | '.join(str(c[k]) for k in ['方面','已处理分母','提及','负面','中性','正面','混合'])+' |')
        lines += ['', '## 待复核记录（不作为可靠证据输出）','']
        for row in records:
            result = selected.get(row['record_id'])
            if result is None or not result['valid'] or result.get('empty_result'):
                lines.append(f"- 来源记录 {row['source_record_id']}："+('待处理' if result is None else result.get('error') or '空结果，需检查遗漏'))
    state = topic_state(folder)
    lines += ["", "## 主题候选", "", "主题来自有效意见子集；未聚类的意见保留为离群项。主题尚未经人工确认，不自动修改正式方面标签。", ""]
    units = {u["unit_id"]: u for u in state["units"] if u["record_id"] in ids}
    for topic in state["topics"]:
        members = [units[uid] for uid, tid in state["assignments"].items() if uid in units and str(tid) == str(topic["id"])]
        if members:
            title = re.sub(r"[\r\n]", " ", topic["name"])
            lines.append(f"- {title}：{len(members)} 个意见，涉及 {len({u['record_id'] for u in members})} 条反馈。")
    lines += ["", "## 原文证据索引", ""]
    for row in records:
        rid = row["record_id"]
        reviewed = reviews.get(rid, {}).get("payload")
        result = valid.get(rid)
        decision = reviewed["decision"] if reviewed else "未人工审核"
        opinions = reviewed["opinions"] if reviewed else result["opinions"] if result else []
        if decision == "rejected" or not opinions:
            continue
        lines += [f"### 来源记录 {row['source_record_id']} · {decision}", "", f"record_id: {rid}", ""]
        if result and result.get("partial"):
            lines += ["本条存在未通过或对象待定的意见；此处只展示已定位内容，不能视为整条审核通过。", ""]
        for op in opinions:
            evidence = op["evidence"].replace("\n", " ").replace("\r", " ")
            # Plain fenced code preserves untrusted Markdown as quoted data.
            quoted_text = evidence + op.get("target", "") + op.get("issue", "")
            fence = "`" * max(3, max((len(m[0])+1 for m in re.finditer(r"`+", quoted_text)), default=3))
            lines += [f"{OPINION_LABELS[op['aspect']]} / {SENTIMENTS[op['sentiment']]} / {op['intent']} · 原文 [{op['start']}, {op['end']})", ""]
            if op.get("scope", "current") != "current":
                lines += [("归属："+{'other':'其他对象','past':'过去体验','uncertain':'待核验'}.get(op['scope'],op['scope']) if candidate else '归属：其他对象或待核验')+"；机器聚类排除此类意见，人工审核不会自动重算旧主题。", ""]
            if op.get("grounding_method") in ("whitespace_restored", "ellipsis_restored"):
                lines += ["引用经过唯一对齐，已恢复为包含中间内容的连续原文；模型原始引用保留在本地运行结果中。", ""]
            if "target" in op:
                lines += ["评价对象（待核验）：", fence, op['target'] or '对象待定', fence, ""]
                if op["issue"] != op["evidence"]:
                    lines += ["意见概述（待核验）：", fence, op["issue"], fence, ""]
            lines += [fence, evidence, fence, ""]
    return "\n".join(lines)
