import json

from echolens.stage3_sft import encode, matching, metrics, parse_output, span_match


def opinion(evidence="好吃", occurrence=1):
    return {"target": "菜", "aspect": "Food#Taste", "sentiment": "positive", "intent": "praise",
            "scope": "current", "evidence": evidence, "occurrence": occurrence}


def test_duplicate_predictions_cannot_match_same_gold_twice():
    answer = json.dumps({"opinions": [opinion()]}, ensure_ascii=False)
    output = json.dumps({"opinions": [opinion(), opinion()]}, ensure_ascii=False)
    score = metrics([{"record_id": "a", "output": output}],
                    [{"record_id": "a", "text": "好吃", "answer": answer}], {"Food#Taste": "口味"})
    assert score["counts"]["strict_tp"] == 1
    assert score["joint_evidence_aspect_sentiment"]["strict"]["precision"] == 0.5


def test_matching_reassigns_greedy_conflict():
    assert len(matching([0, 1], [0, 1], lambda a, b: a == 0 or b == 0)) == 2


def test_evidence_occurrence_and_invalid_label():
    a = {"Food#Taste": "口味"}
    result = parse_output(json.dumps({"opinions": [opinion(occurrence=2)]}), "好吃，好吃", a)
    assert result["valid"][0]["start"] == 3
    result = parse_output(json.dumps({"opinions": [opinion(occurrence=3)]}), "好吃，好吃", a)
    assert result["schema_valid"] and result["errors"] and not result["valid"]
    bad = {**opinion(), "aspect": "invented"}
    assert not parse_output(json.dumps({"opinions": [bad]}), "好吃", a)["schema_valid"]


def test_malformed_json_not_silently_dropped():
    gold = json.dumps({"opinions": [opinion()]})
    result = metrics([{"record_id": "x", "output": '{"opinions":['}],
                     [{"record_id": "x", "text": "好吃", "answer": gold}], {"Food#Taste": "口味"})
    assert result["counts"]["predicted"] == result["counts"]["gold"] == 1
    assert result["counts"]["strict_tp"] == result["json_valid_rate"] == 0


def test_relaxed_span_threshold_and_mask_boundary():
    assert span_match({"start": 0, "end": 4}, {"start": 2, "end": 4}, True)
    assert not span_match({"start": 0, "end": 5}, {"start": 3, "end": 5}, True)
    class Tokenizer:
        eos_token = "!"
        def apply_chat_template(self, *args, **kwargs):
            return "prefix:"
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}
    ids, labels = encode(Tokenizer(), "s", "u", "answer")
    assert labels[:7] == [-100] * 7 and labels[7:] == ids[7:]
