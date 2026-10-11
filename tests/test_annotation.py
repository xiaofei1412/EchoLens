import pytest
from streamlit.testing.v1 import AppTest

from echolens.annotation import (GUIDELINE, digest, load_packet, occurrences, save_annotation,
                                 history, latest, validate_opinions)
from echolens.experiments import sha256
from echolens.workbench import atomic_json


@pytest.fixture
def packet_path(tmp_path):
    text = "🙂服务很好。排队太慢！服务很好。"
    op = {"aspect": "Service#Hospitality", "sentiment": "positive", "intent": "praise",
          "target": "服务", "scope": "current", "evidence": "服务很好。", "start": 1, "end": 6}
    packet = {"schema_version": "opinion-annotation-v1", "guideline_version": "stage3-pilot-v1",
              "guideline_sha256": sha256(GUIDELINE), "synthetic_fixture": True,
              "aspects": {"Service#Hospitality": "服务态度", "unknown": "待定"},
              "records": [{"record_id": str(i), "source_record_id": str(i), "text_raw": text, "suggestions": [op]} for i in range(2)]}
    packet["packet_sha256"] = digest(packet)
    path = tmp_path / "packet.json"
    atomic_json(path, packet)
    return path


def test_packet_freeze_and_exact_repeated_unicode_evidence(packet_path):
    packet = load_packet(packet_path)
    row = packet["records"][0]
    assert occurrences(row["text_raw"], "服务很好。") == [1, 11]
    validate_opinions(row["text_raw"], row["suggestions"], packet["aspects"])
    invalid = {**row["suggestions"][0], "start": 0}
    with pytest.raises(ValueError, match="证据"):
        validate_opinions(row["text_raw"], [invalid], packet["aspects"])
    packet["records"][0]["text_raw"] += "改变"
    atomic_json(packet_path, packet)
    with pytest.raises(ValueError, match="内容已改变"):
        load_packet(packet_path)


def test_history_reopen_and_stale_tab_cannot_overwrite(packet_path):
    packet = load_packet(packet_path)
    folder = packet_path.parent
    ops = packet["records"][0]["suggestions"]
    draft = save_annotation(folder, packet, "0", ops, "draft", "test", origin="automated_fixture")
    done = save_annotation(folder, packet, "0", ops, "completed", "test", coverage=True,
                           expected_revision=draft["revision"], origin="automated_fixture")
    assert latest(folder)["0"] == done
    assert len(history(folder, "0")) == 2
    assert done["training_ready"] is False and done["origin"] == "automated_fixture"
    with pytest.raises(ValueError, match="另一个页面"):
        save_annotation(folder, packet, "0", [], "draft", "test", expected_revision=draft["revision"])
    assert len(history(folder)) == 2


def test_completion_gates_and_unresolved_drafts(packet_path):
    packet = load_packet(packet_path)
    folder = packet_path.parent
    ops = packet["records"][0]["suggestions"]
    with pytest.raises(ValueError, match="已读完整"):
        save_annotation(folder, packet, "0", ops, "completed", "test")
    with pytest.raises(ValueError, match="空意见"):
        save_annotation(folder, packet, "0", [], "completed", "test", coverage=True)
    with pytest.raises(ValueError, match="存在意见"):
        save_annotation(folder, packet, "0", ops, "completed", "test", coverage=True, no_opinions=True)
    with pytest.raises(ValueError, match="待讨论的问题"):
        save_annotation(folder, packet, "0", ops, "needs_discussion", "test")
    invalid = [{**ops[0], "evidence": "非原文", "start": None, "end": None}]
    save_annotation(folder, packet, "0", invalid, "draft", "test", origin="automated_fixture")
    assert latest(folder)["0"]["opinions"] == invalid


def test_unknown_needs_note_and_duplicate_is_rejected(packet_path):
    packet = load_packet(packet_path)
    ops = packet["records"][0]["suggestions"]
    with pytest.raises(ValueError, match="完全重复"):
        save_annotation(packet_path.parent, packet, "0", ops * 2, "completed", "test", coverage=True)
    with pytest.raises(ValueError, match="无法判断项"):
        save_annotation(packet_path.parent, packet, "0", [{**ops[0], "sentiment": "unknown"}], "completed", "test", coverage=True)
    result = save_annotation(packet_path.parent, packet, "0", [], "completed", "test", coverage=True, no_opinions=True, origin="automated_fixture")
    assert result["opinions"] == []


def ui(packet_path):
    script = f"from pathlib import Path\nfrom echolens.annotation_ui import render_annotation_page\nrender_annotation_page(Path({str(packet_path)!r}))"
    return AppTest.from_string(script).run(timeout=20)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_ui_complete_resume_and_backup_without_touching_real_data(packet_path):
    app = ui(packet_path)
    assert not app.exception
    button(app, "保存完整核验并下一条").click().run()
    assert any("已读完整" in e.value for e in app.error)
    app.checkbox[0].check().run()
    button(app, "保存完整核验并下一条").click().run()
    assert not app.exception and "第 2 / 2" in app.subheader[0].value
    assert latest(packet_path.parent)["0"]["origin"] == "automated_fixture"
    reopened = ui(packet_path)
    assert "第 2 / 2" in reopened.subheader[0].value
    button(reopened, "上一条（未保存编辑存为草稿）").click().run()
    assert "第 1 / 2" in reopened.subheader[0].value
    assert reopened.checkbox[0].value is True
    button(reopened, "下一条（未保存编辑存为草稿）").click().run()
    assert latest(packet_path.parent)["0"]["status"] == "completed"


def test_ui_add_copy_delete_merge_and_invalid_quote(packet_path):
    app = ui(packet_path)
    button(app, "复制后拆分").click().run()
    assert len([x for x in app.text_input if x.label == "评价对象"]) == 2
    app.multiselect[0].set_value([0, 1]).run()
    button(app, "合并选中意见").click().run()
    assert len([x for x in app.text_input if x.label == "评价对象"]) == 1
    button(app, "新增空白意见").click().run()
    deletes = [b for b in app.button if b.label == "删除此意见"]
    deletes[1].click().run()
    area = next(x for x in app.text_area if x.label.startswith("原文证据"))
    area.set_value("不存在的引用").run()
    app.checkbox[0].check().run()
    button(app, "保存完整核验并下一条").click().run()
    assert any("证据" in e.value for e in app.error)
    button(app, "保存草稿").click().run()
    assert latest(packet_path.parent)["0"]["opinions"][0]["evidence"] == "不存在的引用"
    reopened = ui(packet_path)
    assert next(x for x in reopened.text_area if x.label.startswith("原文证据")).value == "不存在的引用"
    assert not reopened.exception
