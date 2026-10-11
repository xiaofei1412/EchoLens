import copy
import json
from pathlib import Path
import pytest
from echolens.contracts import validate_opinions

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = json.loads((ROOT / "configs/labels/asap-v0.1.json").read_text(encoding="utf-8"))
VALID = {"opinions": [{"aspect": "Food#Taste", "sentiment": "positive", "intent": "praise", "evidence": "味道很好"}]}


def test_all_synthetic_examples_are_valid():
    for line in (ROOT / "examples/probe.jsonl").read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        validate_opinions({"opinions": sample["opinions"]}, sample["text"], TAXONOMY)


@pytest.mark.parametrize("field,value", [("aspect", "unknown_feature"), ("sentiment", "not_mentioned"), ("intent", "buy"), ("evidence", "味道不太好"), ("evidence", "")])
def test_rejects_invalid_or_fabricated_unit(field, value):
    payload = copy.deepcopy(VALID)
    payload["opinions"][0][field] = value
    with pytest.raises(ValueError):
        validate_opinions(payload, "味道很好", TAXONOMY)


def test_empty_opinions_allowed_when_no_supported_evidence():
    assert validate_opinions({"opinions": []}, "路过打卡", TAXONOMY) == {"opinions": []}


def test_evidence_keeps_negation():
    payload = copy.deepcopy(VALID)
    payload["opinions"][0]["evidence"] = "没有觉得难吃"
    validate_opinions(payload, "没有觉得难吃。", TAXONOMY)
    # Semantic correctness is deliberately NOT guaranteed by substring validation.
