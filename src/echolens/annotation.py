"""Stage 3 pilot annotation: exact evidence and append-only local human edits."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from echolens.experiments import ROOT
from echolens.workbench import read_json

PACKET = ROOT / "data/annotations/stage3-pilot20/packet.json"
GUIDELINE = ROOT / "docs/annotation-stage3.md"
SENTIMENTS = {"positive": "正面", "negative": "负面", "neutral": "中性", "mixed": "混合", "unknown": "无法判断"}
INTENTS = {"praise": "称赞", "complaint": "抱怨", "improvement_request": "请求改善", "help_request": "求助", "statement": "一般陈述", "unknown": "无法判断"}
SCOPES = {"current": "当前对象 / 本次体验", "other": "其他对象 / 竞品", "past": "过去体验 / 旧版本", "uncertain": "归属不确定"}
STATUSES = {"draft": "草稿", "needs_discussion": "待讨论", "completed": "已完整核验"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_packet(path=PACKET):
    packet = read_json(path)
    if not packet:
        raise ValueError("试标包不存在，请先运行 scripts/prepare_stage3_pilot.py")
    content = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if digest(content) != packet["packet_sha256"]:
        raise ValueError("试标包内容已改变，请保留原包并建立新版本")
    return packet


def occurrences(text, evidence):
    if not evidence:
        return []
    return [i for i in range(len(text)) if text.startswith(evidence, i)]


def validate_opinions(text, opinions, aspects):
    seen = set()
    for number, op in enumerate(opinions, 1):
        prefix = f"意见 {number}："
        if op.get("aspect") not in aspects or op.get("sentiment") not in SENTIMENTS or op.get("intent") not in INTENTS or op.get("scope") not in SCOPES:
            raise ValueError(prefix + "标签无效")
        start, end = op.get("start"), op.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != op.get("evidence"):
            raise ValueError(prefix + "证据不是所选位置的原文，请选择片段或粘贴原文")
        if not op.get("target", "").strip():
            raise ValueError(prefix + "请填写评价对象；无法确定时填写“无法确定”并注明归属")
        key = tuple(op[k] for k in ("start", "end", "aspect", "sentiment", "intent", "target", "scope"))
        if key in seen:
            raise ValueError(prefix + "与前面的意见完全重复，请删除重复卡片")
        seen.add(key)


@contextmanager
def connect(folder):
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "annotations.sqlite3")
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, payload TEXT NOT NULL)")
            yield db
    finally:
        db.close()


def history(folder, record_id=None):
    if not (folder / "annotations.sqlite3").exists():
        return []
    with connect(folder) as db:
        query, params = ("SELECT id, payload FROM events ORDER BY id", ()) if record_id is None else ("SELECT id, payload FROM events WHERE record_id=? ORDER BY id", (record_id,))
        return [{**json.loads(raw), "revision": rev} for rev, raw in db.execute(query, params)]


def latest(folder):
    return {event["record_id"]: event for event in history(folder)}


def save_annotation(folder, packet, record_id, opinions, status, actor, note="", coverage=False,
                    no_opinions=False, expected_revision=0, origin="human_ui"):
    row = next(r for r in packet["records"] if r["record_id"] == record_id)
    if status not in STATUSES or not actor.strip():
        raise ValueError("请填写标注者，并选择有效状态")
    if status == "completed":
        if not coverage:
            raise ValueError("请确认已读完整条原文，并检查漏标、对象和重复意见")
        if not opinions and not no_opinions:
            raise ValueError("空意见需要明确确认原文没有适用意见")
        if opinions and no_opinions:
            raise ValueError("已存在意见，不能同时确认没有适用意见")
        validate_opinions(row["text_raw"], opinions, packet["aspects"])
        if any(o["aspect"] == "unknown" or o["sentiment"] == "unknown" or o["intent"] == "unknown" or o["scope"] == "uncertain" or o["target"] == "无法确定" for o in opinions) and not note.strip():
            raise ValueError("存在无法判断项，请说明原因；尚未解决的疑问建议存为待讨论")
    if status == "needs_discussion" and not note.strip():
        raise ValueError("请简单写下待讨论的问题")
    payload = {"record_id": record_id, "packet_sha256": packet["packet_sha256"],
               "guideline_version": packet["guideline_version"], "guideline_sha256": packet["guideline_sha256"],
               "schema_version": packet["schema_version"], "text_sha256": digest(row["text_raw"]),
               "actor": actor.strip(), "origin": origin, "status": status, "opinions": opinions,
               "note": note, "coverage_confirmed": coverage, "no_opinions_confirmed": no_opinions,
               "usage": "pilot_train_only", "training_ready": False,
               "saved_at": datetime.now(timezone.utc).isoformat()}
    with connect(folder) as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT id, payload FROM events WHERE record_id=? ORDER BY id DESC LIMIT 1", (record_id,)).fetchone()
        if (current[0] if current else 0) != expected_revision:
            raise ValueError("另一个页面已保存这条记录，请重新载入后再修改，避免覆盖")
        if current and json.loads(current[1])["packet_sha256"] != packet["packet_sha256"]:
            raise ValueError("审核历史属于不同试标包，不能混写")
        cursor = db.execute("INSERT INTO events(record_id,payload) VALUES (?,?)", (record_id, json.dumps(payload, ensure_ascii=False)))
        payload["revision"] = cursor.lastrowid
    return payload
