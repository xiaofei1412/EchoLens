import hashlib
import json
from pathlib import Path
import pytest
from echolens import workbench as w


def fixture_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "STORE", tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "1", "text": "味道好。上菜太慢！<script>bad</script>"}, ensure_ascii=False), encoding="utf-8")
    dataset_id, _ = w.import_upload(source, {"source_record_id": "id", "text_raw": "text"}, "synthetic", "v1", "synthetic")
    folder = w.dataset_path(dataset_id)
    metadata = w.read_json(folder / "dataset.json")
    metadata["kind"] = "synthetic_fixture"
    w.atomic_json(folder / "dataset.json", metadata)
    return folder, w.records_at(folder)[0]


def test_evidence_unicode_offsets_and_html_escape():
    text = "🙂味道不错。\n  上菜太慢！<script>x</script>"
    output = '{"opinions":[{"a":6,"s":-1,"i":"complaint","e":1}]}'
    op = w.decode_opinions(output, text)[0]
    assert text[op["start"]:op["end"]] == "上菜太慢！"
    rendered = w.highlighted(text, [op])
    assert "<mark>上菜太慢！</mark>" in rendered
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


def test_dataset_snapshot_matches_file_bytes(tmp_path, monkeypatch):
    folder, _ = fixture_dataset(tmp_path, monkeypatch)
    actual = hashlib.sha256((folder / "records.jsonl").read_bytes()).hexdigest()
    assert w.read_json(folder / "dataset.json")["snapshot_sha256"] == actual


@pytest.mark.parametrize("item", [
    {"a":6,"s":-1,"i":"complaint","e":99},
    {"a":True,"s":-1,"i":"complaint","e":0},
    {"a":6,"s":-2,"i":"complaint","e":0},
    {"a":6,"s":-1,"i":"invented","e":0},
])
def test_invalid_generated_labels_and_evidence_are_rejected(item):
    with pytest.raises(ValueError):
        w.decode_opinions(json.dumps({"opinions":[item]}), "上菜慢。")


def test_duplicate_or_over_limit_opinions_rejected():
    item = {"a":6,"s":-1,"i":"complaint","e":0}
    with pytest.raises(ValueError):
        w.decode_opinions(json.dumps({"opinions":[item,item]}), "上菜慢。")
    with pytest.raises(ValueError):
        w.decode_opinions(json.dumps({"opinions":[item]*4}), "上菜慢。")


def test_review_is_append_only_and_export_uses_correction(tmp_path, monkeypatch):
    folder, row = fixture_dataset(tmp_path, monkeypatch)
    rid = row["record_id"]
    original = w.decode_opinions('{"opinions":[{"a":15,"s":1,"i":"praise","e":0}]}', row["text_raw"])
    corrected = w.decode_opinions('{"opinions":[{"a":6,"s":-1,"i":"complaint","e":1}]}', row["text_raw"])
    extractions = {rid: {"valid": True, "opinions": original}}
    w.save_review(folder, rid, corrected, "edited", "fixture", "自动化测试")
    assert extractions[rid]["opinions"] == original
    report = w.report_markdown(folder, [row], {}, extractions)
    assert "上菜太慢" in report and "味道好" not in report
    w.save_review(folder, rid, [], "rejected", "fixture", "自动化测试")
    assert len(w.events(folder)) == 2
    assert "### 来源记录" not in w.report_markdown(folder, [row], {}, extractions)
    assert w.latest_reviews(folder)[rid]["payload"]["origin"] == "automated_fixture"


def test_filtered_denominator_counts_unique_feedbacks():
    rows = [{"record_id": "a"}, {"record_id": "b"}]
    predictions = {"a": {"labels": [-1]*18}, "b": {"labels": [1]*18}}
    counts = w.aspect_counts(rows[:1], predictions)
    assert all(c["已分类分母"] == 1 and c["提及"] == 1 and c["负面"] == 1 for c in counts)


def test_topic_edits_preserve_every_unit_and_history(tmp_path, monkeypatch):
    folder, _ = fixture_dataset(tmp_path, monkeypatch)
    base = {"version": 1, "topics": [{"id":"0","name":"A"},{"id":"1","name":"B"}], "units": [], "assignments": {"a":"0","b":"0","c":"1"}}
    w.atomic_json(folder / "topics.json", base)
    w.edit_topics(folder, "rename", ["0"], "renamed", [], "自动化测试")
    w.edit_topics(folder, "merge", ["0", "1"], "merged", [], "自动化测试")
    w.edit_topics(folder, "split", ["0"], "new", ["a"], "自动化测试")
    state = w.topic_state(folder)
    assert state["version"] == 4 and len(state["topics"]) == 2
    assert set(state["assignments"]) == {"a", "b", "c"}
    assert w.read_json(folder / "topics.json") == base
    assert len(w.events(folder)) == 3


def test_resume_does_not_mutate_analyzed_dataset(tmp_path, monkeypatch):
    folder, _ = fixture_dataset(tmp_path, monkeypatch)
    w.atomic_json(folder / "analysis-run.json", {"locked": True})
    source = tmp_path / "source.jsonl"
    source.write_text('{"id":"2","text":"新反馈"}', encoding="utf-8")
    with pytest.raises(ValueError, match="新版本"):
        w.import_upload(source, {"source_record_id":"id","text_raw":"text"}, "synthetic", "v1", "synthetic")
    assert len(w.records_at(folder)) == 1


def test_context_review_preserves_exact_subclause_and_scope(tmp_path, monkeypatch):
    folder, row = fixture_dataset(tmp_path, monkeypatch)
    text = row["text_raw"]
    start, end = text.index("上菜"), text.index("！")
    op = {"aspect":w.ASPECTS[6], "sentiment":-1, "intent":"complaint",
          "start":start,"end":end,"evidence":text[start:end],"target":"上菜", "issue":"上菜慢", "scope":"uncertain"}
    w.save_review(folder,row["record_id"],[op],"edited","synthetic span check","自动化测试")
    saved = w.latest_reviews(folder)[row["record_id"]]["payload"]["opinions"][0]
    assert (saved["start"],saved["end"],saved["evidence"]) == (start,end,"上菜太慢")
    report = w.report_markdown(folder,[row],{}, {})
    assert "归属：其他对象或待核验" in report and "上菜太慢！" not in report
    with pytest.raises(ValueError,match="归属"):
        w.save_review(folder,row["record_id"],[{**op,"scope":"invented"}],"edited","fixture","自动化测试")
    with pytest.raises(ValueError,match="不能为空"):
        w.save_review(folder,row["record_id"],[{**op,"issue":None}],"edited","fixture","自动化测试")
