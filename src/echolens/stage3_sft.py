"""Frozen SFT prompt, strict output parsing and one-to-one teacher agreement."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTENTS = {"praise", "complaint", "improvement_request", "help_request", "statement", "unknown"}
SCOPES = {"current", "other", "past", "uncertain"}
SENTIMENTS = {"positive", "negative", "neutral", "mixed", "unknown"}


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def occurrences(text, evidence):
    found, start = [], 0
    while evidence:
        pos = text.find(evidence, start)
        if pos < 0:
            break
        found.append(pos)
        start = pos + 1
    return found

FIELDS = ("target", "aspect", "sentiment", "intent", "scope", "evidence", "occurrence")


def system_prompt(aspects):
    return ('完整提取用户反馈中的意见，只输出JSON，不输出思考、Markdown或解释。格式：'
            '{"opinions":[{"target":"具体评价对象","aspect":"方面ID",'
            '"sentiment":"positive/negative/neutral/mixed/unknown",'
            '"intent":"praise/complaint/improvement_request/help_request/statement/unknown",'
            '"scope":"current/other/past/uncertain","evidence":"连续原文片段","occurrence":1}]}。'
            '没有意见则输出空数组。evidence必须逐字连续引用，occurrence为该片段在原文第几次出现，从1开始。'
            '不同对象、子对象、方面分别提取；同对象同方面同态度的重复表述合并。'
            '保留否定、转折、条件和推测，不把数量、尺寸、比较或客服说明自动当负面。'
            '可分离的正负意见分别提取。称赞或最爱不等于向别人推荐；明确菜品推荐才Food#Recommend。'
            'scope的other指竞品/其他对象，past指同对象过去体验，uncertain仅用于对象或时间未定。'
            '具体方面优先于整体评价，不遗漏长文中的子对象评价。方面ID与名称：'
            + json.dumps(aspects, ensure_ascii=False, separators=(",", ":")))


def encode(tokenizer, prompt, text, answer=None):
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if answer is None:
        return ids
    full = tokenizer(prefix + answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    assert full[:len(ids)] == ids, "Tokenizer changed prompt boundary"
    return full, [-100] * len(ids) + full[len(ids):]


def parse_output(output, text, aspects):
    errors, valid = [], []
    try:
        payload = json.loads(output)
        if not isinstance(payload, dict) or set(payload) != {"opinions"} or not isinstance(payload["opinions"], list):
            raise ValueError("Expected opinions array only")
    except (ValueError, TypeError) as exc:
        return {"json_valid": False, "schema_valid": False, "predicted_count": 1,
                "valid": [], "errors": [str(exc)], "evidence_valid_count": 0}
    schema_ok, evidence_count = True, 0
    for op in payload["opinions"]:
        if (not isinstance(op, dict) or set(op) != set(FIELDS)
                or any(not isinstance(op.get(k), str) or not op[k].strip() for k in FIELDS[:-1])
                or type(op.get("occurrence")) is not int
                or op.get("aspect") not in aspects or op.get("sentiment") not in SENTIMENTS
                or op.get("intent") not in INTENTS or op.get("scope") not in SCOPES):
            schema_ok = False
            errors.append("Invalid opinion schema or label")
            continue
        positions = occurrences(text, op["evidence"])
        if not 1 <= op["occurrence"] <= len(positions):
            errors.append("Evidence/occurrence not in original")
            continue
        evidence_count += 1
        start = positions[op["occurrence"] - 1]
        valid.append({**op, "start": start, "end": start + len(op["evidence"])})
    return {"json_valid": True, "schema_valid": schema_ok, "predicted_count": len(payload["opinions"]),
            "valid": valid, "errors": errors, "evidence_valid_count": evidence_count}


def matching(predicted, gold, compatible):
    """Maximum cardinality bipartite matching; each gold opinion counts once."""
    assigned = {}
    def visit(i, seen):
        for j, expected in enumerate(gold):
            if j in seen or not compatible(predicted[i], expected):
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False
    for i in range(len(predicted)):
        visit(i, set())
    return [(i, j) for j, i in assigned.items()]


def span_match(a, b, relaxed=False):
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return (overlap / (max(a["end"], b["end"]) - min(a["start"], b["start"])) >= 0.5
            if relaxed else (a["start"], a["end"]) == (b["start"], b["end"]))


def metrics(results, rows, aspects):
    by_id = {r["record_id"]: r for r in rows}
    totals = {"predicted": 0, "gold": 0, "strict_tp": 0, "relaxed_tp": 0,
              "json_valid": 0, "schema_valid": 0, "evidence_valid": 0, "matched": 0}
    correct = {k: 0 for k in ("aspect", "sentiment", "intent", "scope", "target")}
    for result in results:
        row = by_id[result["record_id"]]
        pred = parse_output(result["output"], row["text"], aspects)
        gold = parse_output(row["answer"], row["text"], aspects)["valid"]
        totals["predicted"] += pred["predicted_count"]
        totals["gold"] += len(gold)
        for k in ("json_valid", "schema_valid"):
            totals[k] += pred[k]
        totals["evidence_valid"] += pred["evidence_valid_count"]
        for mode in ("strict", "relaxed"):
            pairs = matching(pred["valid"], gold, lambda a, b: span_match(a, b, mode == "relaxed")
                             and a["aspect"] == b["aspect"] and a["sentiment"] == b["sentiment"])
            totals[f"{mode}_tp"] += len(pairs)
        pairs = matching(pred["valid"], gold, lambda a, b: span_match(a, b))
        totals["matched"] += len(pairs)
        for i, j in pairs:
            for k in correct:
                correct[k] += pred["valid"][i][k] == gold[j][k]
    score = {}
    for mode in ("strict", "relaxed"):
        tp = totals[f"{mode}_tp"]
        precision, recall = tp / max(totals["predicted"], 1), tp / max(totals["gold"], 1)
        score[mode] = {"precision": precision, "recall": recall,
                       "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    return {"records": len(results), "counts": totals, "joint_evidence_aspect_sentiment": score,
            "json_valid_rate": totals["json_valid"] / max(len(results), 1),
            "schema_valid_rate": totals["schema_valid"] / max(len(results), 1),
            "evidence_valid_rate": totals["evidence_valid"] / max(totals["predicted"], 1),
            "exact_span_matched_field_accuracy": {k: v / max(totals["matched"], 1) for k, v in correct.items()},
            "reference": "AI teacher agreement, not human gold; malformed JSON counts as one false positive",
            "relaxed_definition": "span IoU >= 0.5; both joint scores require aspect and sentiment; one-to-one matching"}
