from copy import deepcopy

import pytest

from echolens.teacher_review import normalize_web_response, validate_response


def sample():
    text = "外皮很脆，馅太咸。"
    inputs = {"aspects": {"Food#Taste": "口味"}, "records": [{"record_id": "r", "text_raw": text,
        "spans": [{"span_id": "s1", "start": 0, "end": 5}, {"span_id": "s2", "start": 5, "end": 9}]}]}
    opinions = [{"opinion_id": "o1", "target": "外皮", "aspect": "Food#Taste", "sentiment": "positive", "intent": "praise", "scope": "current", "evidence": "外皮很脆", "occurrence": 1},
                {"opinion_id": "o2", "target": "馅", "aspect": "Food#Taste", "sentiment": "negative", "intent": "complaint", "scope": "current", "evidence": "馅太咸", "occurrence": 1}]
    coverage = [{"span_id": "s1", "status": "opinion", "opinion_ids": ["o1"], "reason": "外皮口感"}, {"span_id": "s2", "status": "opinion", "opinion_ids": ["o2"], "reason": "馅的味道"}]
    return inputs, {"records": [{"record_id": "r", "opinions": opinions, "coverage": coverage, "review_questions": []}]}


def test_multiple_objects_in_one_sentence_and_machine_provenance():
    inputs, response = sample()
    result = validate_response(response, inputs)
    assert len(result["records"][0]["opinions"]) == 2
    assert result["origin"] == "teacher_model_unreviewed" and not result["training_ready"] and not result["human_gold"]
    assert result["records"][0]["opinions"][1]["start"] == 5


def test_web_empty_references_are_logged_without_changing_semantics_or_source():
    inputs, response = sample()
    row = response["records"][0]
    row["opinions"].pop()
    row["coverage"][1] = {"span_id": "s2", "status": "uncertain", "reason": "fixture uncertainty"}
    original = deepcopy(response)
    normalized, repairs = normalize_web_response(response)
    assert response == original and len(repairs) == 1
    assert normalized["records"][0]["coverage"][1]["opinion_ids"] == []
    assert normalized["records"][0]["opinions"] == response["records"][0]["opinions"]
    validate_response(normalized, inputs)


@pytest.mark.parametrize("kind", ["missing_opinion_reference", "explicit_null", "wrong_type", "missing_status"])
def test_web_normalizer_does_not_invent_nonempty_references(kind):
    inputs, response = sample()
    entry = response["records"][0]["coverage"][0]
    if kind == "missing_opinion_reference": del entry["opinion_ids"]
    if kind == "explicit_null": entry.update(status="context", opinion_ids=None)
    if kind == "wrong_type": entry.update(status="context", opinion_ids="o1")
    if kind == "missing_status": del entry["status"]
    normalized, repairs = normalize_web_response(response)
    assert repairs == []
    with pytest.raises(ValueError): validate_response(normalized, inputs)


@pytest.mark.parametrize("fault", ["missing_record", "duplicate_record", "missing_span", "bad_reference", "wrong_span", "invented_quote", "bad_occurrence", "duplicate_opinion"])
def test_incomplete_and_fabricated_responses_rejected(fault):
    inputs, response = sample()
    r = response["records"][0]
    if fault == "missing_record": response["records"] = []
    if fault == "duplicate_record": response["records"].append(deepcopy(r))
    if fault == "missing_span": r["coverage"].pop()
    if fault == "bad_reference": r["coverage"][0]["opinion_ids"] = ["o3"]
    if fault == "wrong_span": r["coverage"][0]["opinion_ids"] = ["o2"]
    if fault == "invented_quote": r["opinions"][0]["evidence"] = "很好吃"
    if fault == "bad_occurrence": r["opinions"][0]["occurrence"] = True
    if fault == "duplicate_opinion": r["opinions"].append({**r["opinions"][0], "opinion_id": "o3"})
    with pytest.raises(ValueError): validate_response(response, inputs)


def test_opposite_object_overlap_is_warning_not_silently_merged():
    inputs, response = sample()
    response["records"][0]["opinions"][0]["evidence"] = inputs["records"][0]["text_raw"]
    result = validate_response(response, inputs)
    assert len(result["warnings"]) == 1
    assert len(result["records"][0]["opinions"]) == 2


def test_checker_builds_changes_only_report_without_changing_reference(tmp_path, monkeypatch):
    import importlib.util
    import json
    from echolens.experiments import ROOT, sha256
    from echolens.workbench import atomic_json

    spec = importlib.util.spec_from_file_location("check_stage3_teacher", ROOT / "scripts/check_stage3_teacher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "data/annotations/stage3-teacher-pilot-v1"
    inputs, response = sample()
    inputs["records"][0]["source_record_id"] = "synthetic"
    response["records"][0]["opinions"][0]["target"] = "<script>bad</script>"
    atomic_json(folder / "inputs.json", inputs)
    atomic_json(folder / "human-reference-DO-NOT-UPLOAD.json", {"r": {"opinions": []}})
    hashes = {name: sha256(folder / name) for name in ("inputs.json", "human-reference-DO-NOT-UPLOAD.json")}
    atomic_json(folder / "manifest.json", {"files": hashes})
    response_path = tmp_path / "fixture-response.json"
    atomic_json(response_path, response)
    monkeypatch.setattr("sys.argv", ["check", str(response_path), "--model", "synthetic-test", "--source-id", "synthetic"])
    module.main()
    output = next((folder / "responses").glob("*/review-diff.html")).read_text(encoding="utf-8")
    assert "&lt;script&gt;bad&lt;/script&gt;" in output and "<script>bad</script>" not in output
    assert all(sha256(folder / name) == checksum for name, checksum in hashes.items())
    verified = json.loads(next((folder / "responses").glob("*/validated-response.json")).read_text(encoding="utf-8"))
    assert verified["provenance"]["model_user_reported"] == "synthetic-test"
    assert verified["human_gold"] is False
