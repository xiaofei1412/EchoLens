"""Import an authorized CSV/JSONL using an explicit JSON field mapping."""
import argparse
import json
from pathlib import Path
import duckdb
from echolens.ingest import read_records, import_records, quality_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--rights", default="unknown")
    parser.add_argument("--database", type=Path, default=Path("data/imports/feedback.duckdb"))
    args = parser.parse_args()
    records, stats = read_records(args.file, mapping=json.loads(args.mapping.read_text(encoding="utf-8")), source=args.source, revision=args.revision, rights_status=args.rights)
    args.database.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.database)) as connection:
        stats["inserted"] = import_records(connection, records)
        stats["quality"] = quality_summary(records)
        connection.sql("SELECT * FROM feedback ORDER BY record_id").write_parquet(str(args.database.with_suffix(".parquet")), compression="zstd")
    args.database.with_suffix(".last-import.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
