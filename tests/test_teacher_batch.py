import importlib.util
from copy import deepcopy

import pytest

from echolens.experiments import ROOT, sha256
from echolens.workbench import atomic_json, read_json
from test_teacher_review import sample

spec = importlib.util.spec_from_file_location("check_stage3_batch", ROOT / "scripts/check_stage3_batch.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def fixture_task(folder):
    inputs, response = sample()
    (folder / "rules.md").write_text("Synthetic test only", encoding="utf-8")
    atomic_json(folder / "inputs/batch-0001.json", inputs)
    atomic_json(folder / "outputs/batch-0001.r1.json", response)
    atomic_json(folder / "manifest.json", {"files": {p: sha256(folder / p) for p in ("rules.md", "inputs/batch-0001.json")},
                "batches": {"batch-0001": {"records": 1}}})
    return inputs, response


def test_checked_batch_retains_raw_machine_origin_and_resumes(tmp_path):
    fixture_task(tmp_path)
    first = checker.check_batch(tmp_path, "batch-0001", 1, "synthetic-test", "unknown")
    receipt = tmp_path / "checks/batch-0001.r1.json"
    before = receipt.read_bytes()
    assert checker.check_batch(tmp_path, "batch-0001", 1, "synthetic-test", "unknown") == first
    assert receipt.read_bytes() == before
    result = read_json(receipt)
    assert result["origin"] == "teacher_model_unreviewed"
    assert not result["training_ready"] and not result["human_gold"]
    assert not result["model_identity_verified"]
    raw = tmp_path / "checks/raw" / (result["response_sha256"] + ".json")
    assert raw.read_bytes() == (tmp_path / "outputs/batch-0001.r1.json").read_bytes()
    assert checker.status(tmp_path)["pending_batches"] == []


@pytest.mark.parametrize("fault", ["missing_record", "wrong_batch_id", "fabricated_quote", "omitted_empty_refs"])
def test_invalid_batch_cannot_get_receipt(tmp_path, fault):
    _, response = fixture_task(tmp_path)
    if fault == "missing_record": response["records"] = []
    if fault == "wrong_batch_id": response["records"][0]["record_id"] = "other"
    if fault == "fabricated_quote": response["records"][0]["opinions"][0]["evidence"] = "Invented"
    if fault == "omitted_empty_refs":
        response["records"][0]["coverage"][1] = {"span_id": "s2", "status": "no_opinion", "reason": "fixture"}
    atomic_json(tmp_path / "outputs/batch-0001.r1.json", response)
    with pytest.raises(ValueError): checker.check_batch(tmp_path, "batch-0001", 1, "synthetic-test", "unknown")
    assert not (tmp_path / "checks/batch-0001.r1.json").exists()


@pytest.mark.parametrize("changed_file", ["rules.md", "inputs/batch-0001.json"])
def test_frozen_inputs_cannot_drift(tmp_path, changed_file):
    fixture_task(tmp_path)
    with (tmp_path / changed_file).open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Frozen task changed"):
        checker.check_batch(tmp_path, "batch-0001", 1, "synthetic-test", "unknown")


def test_modified_checked_attempt_rejected_and_revision_preserved(tmp_path):
    _, response = fixture_task(tmp_path)
    checker.check_batch(tmp_path, "batch-0001", 1, "synthetic-test", "unknown")
    changed = deepcopy(response)
    changed["records"][0]["review_questions"] = ["Synthetic ambiguity"]
    atomic_json(tmp_path / "outputs/batch-0001.r1.json", changed)
    assert checker.status(tmp_path)["stale_attempts"] == ["batch-0001.r1.json"]
    with pytest.raises(ValueError, match="Previously checked attempt changed"):
        checker.check_batch(tmp_path, "batch-0001", 1, "synthetic-test", "unknown")
    atomic_json(tmp_path / "outputs/batch-0001.r1.json", response)
    atomic_json(tmp_path / "outputs/batch-0001.r2.json", changed)
    assert checker.check_batch(tmp_path, "batch-0001", 2, "synthetic-test", "unknown")["review_question_records"] == 1
    assert len(list((tmp_path / "checks").glob("batch-*.json"))) == 2
    assert checker.status(tmp_path)["stale_attempts"] == []
