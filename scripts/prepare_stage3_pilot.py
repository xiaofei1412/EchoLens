"""Freeze 20 train-only annotation tasks; reuse predictions without new inference."""
import random
from collections import Counter

from echolens.annotation import PACKET, GUIDELINE, digest, load_packet
from echolens.experiments import ROOT, sha256
from echolens.workbench import dataset_path, records_at, read_json, atomic_json, OPINION_LABELS


def main():
    if PACKET.exists():
        packet = load_packet()
        print(f"Preserved existing packet: {len(packet['records'])} records; {packet['packet_sha256']}")
        return
    source = dataset_path("asap-demo-1000-v3")
    rows = records_at(source)
    if len(rows) != 1000 or any(r["split"] != "train" or not r["clean_keep"] for r in rows):
        raise ValueError("Pilot requires the clean train-only stage 2 snapshot")
    cases = {"5365", "38689", "6479", "44572", "32524", "22304", "26480", "44206", "33803", "15748"}
    difficult = [r for r in rows if r["source_record_id"] in cases]
    assert len(difficult) == 10
    random_rows = random.Random(20261009).sample([r for r in rows if r["source_record_id"] not in cases], 10)
    selected = sorted(random_rows, key=lambda r: len(r["text_raw"])) + sorted(difficult, key=lambda r: len(r["text_raw"]))
    records, hashes = [], {}
    for row in selected:
        result_path = source / "extractions" / f"{row['record_id']}.json"
        result = read_json(result_path, {})
        suggestions = []
        for op in result.get("opinions", []):
            suggestions.append({k: op.get(k, "") for k in ("aspect", "intent", "target", "scope", "evidence", "start", "end")})
            suggestions[-1]["sentiment"] = {-1: "negative", 0: "neutral", 1: "positive"}.get(op["sentiment"], "unknown")
        hashes[row["record_id"]] = sha256(result_path)
        records.append({k: row[k] for k in ("record_id", "source_record_id", "text_raw", "source", "source_revision", "rights_status", "split", "duplicate_group_id")})
        records[-1].update({"selection": "known_difficulty" if row["source_record_id"] in cases else "seeded_random",
                            "suggestions": suggestions, "suggestion_origin": "stage2_v3_machine_unreviewed"})
    assert len({r["duplicate_group_id"] for r in records}) == 20
    packet = {"schema_version": "opinion-annotation-v1", "guideline_version": "stage3-pilot-v1",
              "guideline_sha256": sha256(GUIDELINE), "seed": 20261009, "usage": "pilot_train_only",
              "not_gold": True, "aspects": OPINION_LABELS, "records": records,
              "source": {"dataset": source.name, "records_sha256": sha256(source / "records.jsonl"),
                         "run_sha256": sha256(source / "analysis-run.json"),
                         "models": read_json(source / "analysis-run.json")["models"], "prediction_sha256": hashes}}
    packet["packet_sha256"] = digest(packet)
    atomic_json(PACKET, packet)
    report = {"packet_sha256": packet["packet_sha256"], "records": len(records), "seed": packet["seed"],
              "selection": dict(Counter(r["selection"] for r in records)), "split": "clean-train only",
              "machine_suggested_opinions": sum(len(r["suggestions"]) for r in records),
              "human_completed_at_creation": 0, "usage": "pilot_train_only; not evaluation or training-ready",
              "guideline_sha256": packet["guideline_sha256"], "source": {k: v for k, v in packet["source"].items() if k != "prediction_sha256"}}
    atomic_json(ROOT / "docs/reports/stage3-pilot-preparation.json", report)
    print(f"Prepared {len(records)} records; {report['machine_suggested_opinions']} machine suggestions; no human labels")


if __name__ == "__main__":
    main()
