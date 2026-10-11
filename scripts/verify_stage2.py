"""Verify the completed 1,000-record workflow and export local/public reports."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import duckdb
from echolens.experiments import ROOT, sha256, save_json
from echolens.workbench import (dataset_path, records_at, results_at, read_json, highlighted,
                               aspect_counts, topic_state, report_markdown, latest_reviews)


def main():
    folder = dataset_path("asap-demo-1000")
    records = records_at(folder)
    metadata = read_json(folder / "dataset.json")
    assert len(records) == len({r["record_id"] for r in records}) == 1000
    assert all(r["split"] == "train" and r["clean_keep"] for r in records)
    assert metadata["snapshot_sha256"] == sha256(folder / "records.jsonl")
    classifications = read_json(folder / "classifications.json")
    extractions = results_at(folder)
    ids = {r["record_id"] for r in records}
    assert set(classifications) == set(extractions) == ids
    by_id = {r["record_id"]: r for r in records}
    valid = [r for r in extractions.values() if r["valid"]]
    for result in valid:
        highlighted(by_id[result["record_id"]]["text_raw"], result["opinions"])
        assert len(result["opinions"]) <= 3
    state = topic_state(folder)
    expected_units = {f"{r['record_id']}:{i}" for r in valid for i in range(len(r["opinions"]))}
    assert {u["unit_id"] for u in state["units"]} == set(state["assignments"]) == expected_units
    assert all(str(tid) in {str(t["id"]) for t in state["topics"]} | {"-1"} for tid in state["assignments"].values())
    # Independent SQL aggregation checks the Python display/report counts.
    with duckdb.connect(str(folder / "feedback.duckdb"), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1000
    with duckdb.connect() as con:
        con.execute("CREATE TABLE predictions(labels INTEGER[])")
        con.executemany("INSERT INTO predictions VALUES (?)", [(r["labels"],) for r in classifications.values()])
        for i, counts in enumerate(aspect_counts(records, classifications), 1):
            values = con.execute(f"SELECT count(*),count(*) FILTER(WHERE labels[{i}] != -2),count(*) FILTER(WHERE labels[{i}] = -1),count(*) FILTER(WHERE labels[{i}] = 0),count(*) FILTER(WHERE labels[{i}] = 1) FROM predictions").fetchone()
            assert values == tuple(counts[k] for k in ("已分类分母","提及","负面","中性","正面"))
    report = report_markdown(folder, records, classifications, extractions)
    local = ROOT / "artifacts/stage-02/feedback-report.md"
    local.write_text(report, encoding="utf-8", newline="\n")
    result = {"dataset": metadata["dataset_id"], "snapshot_sha256": metadata["snapshot_sha256"],
              "rows": len(records), "classified": len(classifications), "extraction_attempted": len(extractions),
              "structure_and_evidence_valid": len(valid), "valid_rate": len(valid)/len(records),
              "opinions": len(expected_units), "all_accepted_evidence_spans_match": True,
              "failures": dict(Counter(r["error"] for r in extractions.values() if not r["valid"])),
              "topics": len(state["topics"]), "outlier_opinions": sum(str(t)=="-1" for t in state["assignments"].values()),
              "topic_sizes": dict(Counter(str(t) for t in state["assignments"].values() if str(t) != "-1")),
              "topic_pipeline_seconds": state.get("seconds"),
              "naming_methods": dict(Counter(t.get("name_source", "unknown") for t in state["topics"])),
              "human_reviewed_records": len(latest_reviews(folder)),
              "generation_batch_seconds_allocated_to_rows": sum(r["seconds"] for r in extractions.values()),
              "python_sql_counts_equal": True, "local_report_sha256": hashlib.sha256(report.encode()).hexdigest(),
              "limitations": ["Train-only workflow demonstration, not held-out accuracy", "Sentence-ID grounding guarantees position existence, not semantic support", "At most 3 opinions per review, incomplete recall", "Failures retained for review; no silent output truncation", "Topic names and representatives are machine candidates", "One cluster contains most opinions across multiple aspects; topic quality is not release-ready"]}
    save_json(ROOT / "docs/reports/stage2-workflow.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
