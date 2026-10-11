"""Audit official splits and select train-only exploration records."""
import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    # Whitespace-only duplicate check, NOT the model input cleaning policy.
    return re.sub(r"\s+", "", text)


def main():
    config = json.loads((ROOT / "configs/asap.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "configs/labels/asap-v0.1.json").read_text(encoding="utf-8"))["aspects"]
    source = ROOT / "data/raw/asap" / config["revision"]
    groups = defaultdict(list)
    all_rows = {}
    report = {"revision": config["revision"], "splits": {}, "limitations": ["No time/product/author IDs", "Whitespace normalization is not semantic near-duplicate detection"]}
    for split, expected in config["expected_rows"].items():
        with (source / f"data/{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        assert fields == ["id", "review", "star", *labels], fields
        assert len(rows) == expected, (split, len(rows), expected)
        assert all(row["review"].strip() for row in rows), "Empty review"
        assert all(int(row[a]) in (-2, -1, 0, 1) for row in rows for a in labels), "Invalid label"
        assert all(float(row["star"]).is_integer() and 1 <= float(row["star"]) <= 5 for row in rows), "Invalid star rating"
        for row in rows:
            row["star"] = str(int(float(row["star"])))
        counts = {a: dict(Counter(row[a] for row in rows)) for a in labels}
        lengths = sorted(len(row["review"]) for row in rows)
        for row in rows:
            digest = hashlib.sha256(normalized(row["review"]).encode()).hexdigest()
            groups[digest].append({"split": split, "id": row["id"]})
        report["splits"][split] = {
            "rows": len(rows), "fields": fields, "unique_ids": len({r["id"] for r in rows}),
            "stars": dict(Counter(row["star"] for row in rows)), "aspect_labels": counts,
            "length_chars": {"min": min(lengths), "median": statistics.median(lengths), "p95": lengths[int(.95 * (len(lengths) - 1))], "max": max(lengths)},
            "exact_duplicate_excess": len(rows) - len({r["review"] for r in rows}),
            "all_aspects_not_mentioned": sum(all(r[a] == "-2" for a in labels) for r in rows),
            "literal_escaped_newline_reviews": sum("\\n" in r["review"] or "\\r" in r["review"] for r in rows),
        }
        all_rows[split] = rows
    cross_split = [v for v in groups.values() if len({r["split"] for r in v}) > 1]
    report["cross_split_whitespace_duplicate_groups"] = cross_split
    report["within_or_cross_split_whitespace_duplicate_groups"] = sum(len(v) > 1 for v in groups.values())
    report["all_unique_ids"] = len({r["id"] for rows in all_rows.values() for r in rows})
    # Exclude any train record sharing normalized text with dev/test from exploration.
    held_out = {normalized(r["review"]) for s in ("dev", "test") for r in all_rows[s]}
    eligible = [r for r in all_rows["train"] if normalized(r["review"]) not in held_out]
    # Stratified by star rating, for exploration only, not population estimates.
    rng = random.Random(config["sample_seed"])
    sample = []
    for star in range(1, 6):
        pool = [r for r in eligible if r["star"] == str(star)]
        sample.extend(rng.sample(pool, config["sample_size"] // 5))
    rng.shuffle(sample)
    sample_dir = ROOT / "data/exploration"
    sample_dir.mkdir(parents=True, exist_ok=True)
    with (sample_dir / "asap-200.jsonl").open("w", encoding="utf-8") as out:
        for index, row in enumerate(sample, 1):
            item = {"sample_index": index, "split": "train", "source_id": row["id"], "text": row["review"], "star": int(row["star"]), "aspects": {a: int(row[a]) for a in labels if row[a] != "-2"}, "review_status": "awaiting_ai_reading_then_human_review"}
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    report["exploration"] = {"size": len(sample), "seed": config["sample_seed"], "strategy": "40 per star; train only; excludes dev/test normalized duplicates", "sample_ids": [r["id"] for r in sample]}
    report_dir = ROOT / "docs/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "asap-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": {s: v["rows"] for s, v in report["splits"].items()}, "cross_split_duplicate_groups": len(cross_split), "sample_size": len(sample)}, indent=2))


if __name__ == "__main__":
    main()
