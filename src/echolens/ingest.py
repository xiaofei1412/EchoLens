"""Local, versioned CSV/JSONL ingestion. Raw text is never overwritten."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import duckdb


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def read_records(path, *, mapping, source, revision, rights_status="unknown", split=None, aspects=()):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        raise ValueError("Only CSV and JSONL are supported")
    result, rejected = {}, []
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for line, row in enumerate(rows, 1):
        text = row.get(mapping["text_raw"])
        sid = row.get(mapping["source_record_id"])
        if not isinstance(text, str) or not text.strip() or sid is None or not str(sid).strip():
            rejected.append({"row": line, "reason": "empty_text_or_id"})
            continue
        rating = row.get(mapping.get("rating", ""))
        if rating is not None and rating != "":
            rating = float(rating)
            if not rating.is_integer() or not 1 <= rating <= 5:
                raise ValueError(f"Invalid 1-5 rating at row {line}")
            rating = int(rating)
        else:
            rating = None
        labels = [int(row[a]) for a in aspects] if aspects else None
        if labels is not None and any(v not in (-2, -1, 0, 1) for v in labels):
            raise ValueError(f"Invalid aspect label at row {line}")
        record_id = digest(json.dumps([source, revision, str(sid)], ensure_ascii=False))
        item = {"record_id": record_id, "source": source, "source_revision": revision,
                "source_record_id": str(sid), "text_raw": text, "text_normalized": normalize(text),
                "rating": rating, "split": split, "labels": labels, "rights_status": rights_status,
                "language": "zh_heuristic" if re.search(r"[\u3400-\u9fff]", text) else "und",
                "source_file_sha256": file_hash, "processing_version": "normalize-v1"}
        for field in ("product_id", "created_at", "collected_at", "parent_id", "source_url"):
            value = row.get(mapping.get(field, ""))
            item[field] = str(value) if value not in (None, "") else None
        if record_id in result and item != result[record_id]:
            raise ValueError(f"Conflicting duplicate ID at row {line}; use a new source revision")
        result[record_id] = item
    return list(result.values()), {"input_rows": len(rows), "valid_unique_ids": len(result),
                                "rejected": rejected, "repeated_id_rows": len(rows) - len(rejected) - len(result)}


def import_records(connection, records):
    connection.execute("""CREATE TABLE IF NOT EXISTS feedback (
        record_id VARCHAR PRIMARY KEY, source VARCHAR, source_revision VARCHAR, source_record_id VARCHAR,
        text_raw VARCHAR, text_normalized VARCHAR, rating INTEGER, split VARCHAR, labels INTEGER[],
        rights_status VARCHAR, language VARCHAR, source_file_sha256 VARCHAR, processing_version VARCHAR,
        product_id VARCHAR, created_at VARCHAR, collected_at VARCHAR, parent_id VARCHAR, source_url VARCHAR
    )""")
    if not records:
        return 0
    # A relation from explicit JSON avoids pandas/Arrow and bulk INSERT overhead on Windows.
    payload = json.dumps(records, ensure_ascii=False)
    connection.execute("CREATE OR REPLACE TEMP TABLE incoming AS SELECT unnest(?::JSON[]) AS item", [payload])
    columns = [r[1] for r in connection.execute("PRAGMA table_info('feedback')").fetchall()]
    expressions = [f"item->>'{c}' AS {c}" for c in columns]
    expressions[columns.index("labels")] = "(item->'labels')::INTEGER[] AS labels"
    connection.execute("CREATE OR REPLACE TEMP VIEW incoming_rows AS SELECT " + ",".join(expressions) + " FROM incoming")
    # Versioned source is immutable: retry is allowed, changed content is not silently overwritten.
    differences = " OR ".join(f"CAST(f.{c} AS VARCHAR) IS DISTINCT FROM CAST(i.{c} AS VARCHAR)" for c in columns if c not in ("record_id", "source_file_sha256"))
    if connection.execute(f"SELECT count(*) FROM feedback f JOIN incoming_rows i USING(record_id) WHERE {differences}").fetchone()[0]:
        raise ValueError("Existing record changed; assign a new source revision")
    before = connection.execute("SELECT count(*) FROM feedback").fetchone()[0]
    connection.execute("INSERT INTO feedback BY NAME SELECT * FROM incoming_rows ON CONFLICT DO NOTHING")
    return connection.execute("SELECT count(*) FROM feedback").fetchone()[0] - before


def quality_summary(records):
    return {"rows": len(records), "languages": dict(Counter(r["language"] for r in records)),
            "missing": {key: sum(r[key] is None for r in records) for key in
                        ("rating", "product_id", "created_at", "collected_at", "parent_id", "source_url")},
            "normalization_changed": sum(r["text_raw"] != r["text_normalized"] for r in records),
            "rights": dict(Counter(r["rights_status"] for r in records)),
            "note": "Language uses Han-character presence, not a multilingual language detector"}
