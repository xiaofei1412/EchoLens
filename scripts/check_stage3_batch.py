"""Validate a local teacher batch; never promote it to human gold."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from echolens.experiments import ROOT, sha256
from echolens.teacher_review import validate_response
from echolens.workbench import atomic_json, read_json

TASK = ROOT / "data/annotations/stage3-teacher-900-v2"


def load_task(folder):
    manifest = read_json(folder / "manifest.json")
    for name, checksum in manifest["files"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Frozen task changed: {name}; create a new task version")
    return manifest


def check_batch(folder, batch, attempt, model, thinking):
    manifest = load_task(folder)
    if batch not in manifest["batches"] or not re.fullmatch(r"[1-9][0-9]*", str(attempt)):
        raise ValueError("Unknown batch or invalid attempt")
    stem = f"{batch}.r{attempt}"
    source = folder / "outputs" / f"{stem}.json"
    receipt = folder / "checks" / f"{stem}.json"
    checksum = sha256(source)
    previous = read_json(receipt)
    if previous and (previous["response_sha256"] != checksum or previous["declared_model"] != model
                     or previous["declared_thinking"] != thinking):
        raise ValueError("Previously checked attempt changed; restore it and use a new revision")
    inputs = read_json(folder / "inputs" / f"{batch}.json")
    # Strict format for file-writing agents: unlike legacy web imports, no repairs.
    result = validate_response(read_json(source), inputs)
    result.update(batch=batch, attempt=attempt, declared_model=model, declared_thinking=thinking,
                  model_identity_verified=False, provider="Codex (agent-declared)",
                  response_sha256=checksum, input_sha256=sha256(folder / "inputs" / f"{batch}.json"),
                  manifest_sha256=sha256(folder / "manifest.json"),
                  rule_sha256=manifest["files"]["rules.md"],
                  checked_at=datetime.now(timezone.utc).isoformat(),
                  review_question_records=sum(bool(r["review_questions"]) for r in result["records"]),
                  unknown_aspects=sum(o["aspect"] == "unknown" for r in result["records"] for o in r["opinions"]))
    if not previous:
        # Keep a content-addressed copy, even if someone later edits the output file.
        archive = folder / "checks" / "raw" / f"{checksum}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(source.read_bytes())
        atomic_json(receipt, result)
    return {"batch": batch, "attempt": attempt, "records": len(result["records"]),
            "opinions": sum(len(r["opinions"]) for r in result["records"]),
            "warnings": len(result["warnings"]), "review_question_records": result["review_question_records"],
            "unknown_aspects": result["unknown_aspects"], "human_gold": False, "training_ready": False}


def status(folder):
    manifest = load_task(folder)
    checked, stale = set(), []
    for path in sorted((folder / "checks").glob("batch-*.r*.json")):
        receipt = read_json(path)
        source = folder / "outputs" / path.name
        if (not source.exists() or sha256(source) != receipt["response_sha256"]
                or sha256(folder / "manifest.json") != receipt["manifest_sha256"]):
            stale.append(path.name)
        else:
            checked.add(receipt["batch"])
    return {"checked_batches": sorted(checked), "pending_batches": [b for b in manifest["batches"] if b not in checked],
            "stale_attempts": stale, "semantic_approval": False, "human_gold": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=TASK)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--batch")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--model", help="Actual UI model if known; declaration, not verified backend identity")
    parser.add_argument("--thinking", default="unknown")
    args = parser.parse_args()
    if args.status:
        result = status(args.task_dir)
    else:
        if not args.batch or not args.model:
            parser.error("--batch and --model are required unless --status")
        result = check_batch(args.task_dir, args.batch, args.attempt, args.model, args.thinking)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
