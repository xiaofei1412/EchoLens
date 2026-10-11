import json
import duckdb
import numpy as np
import pytest
from echolens.ingest import read_records, import_records, normalize
from echolens.evaluate import evaluate
from echolens.quality import duplicate_groups, clean_membership


def test_import_idempotence_and_no_silent_overwrite(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"id": "1", "body": "不好吃！\r\n但态度好🙂"}, ensure_ascii=False) + "\n", encoding="utf-8")
    rows, stats = read_records(path, mapping={"source_record_id": "id", "text_raw": "body"}, source="synthetic", revision="v1")
    with duckdb.connect() as con:
        assert import_records(con, rows) == 1
        assert import_records(con, rows) == 0
        assert con.execute("SELECT created_at, product_id, rights_status FROM feedback").fetchone() == (None, None, "unknown")
        rows[0]["text_raw"] = "changed"
        with pytest.raises(ValueError, match="changed"):
            import_records(con, rows)
        assert con.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1


def test_csv_rejects_empty_and_preserves_negation(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text('id,body\n1,不好吃！🙂\n2,\n', encoding="utf-8")
    rows, stats = read_records(path, mapping={"source_record_id": "id", "text_raw": "body"}, source="synthetic", revision="v1")
    assert len(rows) == 1 and stats["rejected"] == [{"row": 2, "reason": "empty_text_or_id"}]
    assert normalize(rows[0]["text_raw"]) == "不好吃！🙂"
    assert rows[0]["created_at"] is None


def test_same_record_across_csv_and_jsonl_is_not_duplicated(tmp_path):
    csv_path, json_path = tmp_path / "data.csv", tmp_path / "data.jsonl"
    csv_path.write_text('id,text\n1,不是很差\n', encoding="utf-8")
    json_path.write_text(json.dumps({"id": "1", "text": "不是很差"}, ensure_ascii=False), encoding="utf-8")
    with duckdb.connect() as con:
        counts = []
        for path in (csv_path, json_path):
            rows, _ = read_records(path, mapping={"source_record_id": "id", "text_raw": "text"}, source="fixture", revision="v1")
            counts.append(import_records(con, rows))
        assert counts == [1, 0]


def test_missed_mentions_are_not_removed_from_sentiment_denominator():
    t = np.array([[-1, -2], [1, 0]])
    y = np.array([[-2, -2], [1, -2]])
    m = evaluate(t, y, ["a", "b"])
    assert m["sentiment_on_gold_mentions"]["positions"] == 3
    assert m["sentiment_on_gold_mentions"]["accuracy"] == pytest.approx(1/3)
    assert m["aspects"]["micro_f1"] == pytest.approx(0.5)
    assert m["pairs"]["micro_f1"] == pytest.approx(0.5)


def test_wrong_sentiment_is_fp_and_fn_for_pair_but_correct_aspect():
    m = evaluate([[-1, -2]], [[1, -2]], ["a", "b"])
    assert m["aspects"]["micro_f1"] == 1
    assert m["pairs"]["micro_f1"] == 0
    assert m["pairs"]["per_label"]["a/-1"]["support"] == 1


def test_duplicate_groups_separate_splits_without_changing_labels():
    rows = [{"record_id": str(i), "source_record_id": str(i), "split": s, "text_raw": text} for i, (s, text) in enumerate([
        ("train", "味道不好服务好" * 20), ("test", "味道不好 服务好" * 20), ("dev", "环境安静干净")])]
    cfg = {"num_perm": 128, "lsh_threshold": .7, "char_ngram": 5, "min_chars": 50, "verified_jaccard": .9}
    groups, pairs = duplicate_groups(rows, cfg, 42)
    assert groups["0"] == groups["1"] != groups["2"]
    assert clean_membership(rows, groups) == {"1", "2"}
    assert pairs[0]["kind"] == "whitespace_exact"


def test_near_duplicate_pair_is_verified_and_split_isolated():
    text = "".join(chr(0x4e00 + i) for i in range(160))
    rows = [{"record_id": "a", "source_record_id": "a", "split": "train", "text_raw": text},
            {"record_id": "b", "source_record_id": "b", "split": "dev", "text_raw": text + "。"}]
    cfg = {"num_perm": 128, "lsh_threshold": .7, "char_ngram": 5, "min_chars": 50, "verified_jaccard": .9}
    groups, pairs = duplicate_groups(rows, cfg, 42)
    assert pairs[0]["kind"] == "char_ngram_jaccard"
    assert pairs[0]["jaccard"] >= .9
    assert clean_membership(rows, groups) == {"b"}


def test_prepared_snapshots_match_manifest_and_clean_groups_do_not_overlap():
    from echolens.experiments import ROOT, sha256
    folder = ROOT / "data/processed/stage1"
    if not (folder / "manifest.json").exists():
        pytest.skip("Prepare full dataset first")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert sha256(folder / "records.jsonl") == manifest["snapshot_sha256"]
    with duckdb.connect() as con:
        groups = set()
        for split in ("train", "dev", "test"):
            official = con.execute("SELECT record_id, labels FROM read_parquet(?)", [str(folder / f"official-{split}.parquet")]).fetchall()
            clean = con.execute("SELECT record_id, labels, duplicate_group_id FROM read_parquet(?)", [str(folder / f"clean-{split}.parquet")]).fetchall()
            source_labels = dict(official)
            assert len(official) == manifest["official_rows"][split]
            assert len(clean) == manifest["clean_rows"][split]
            for rid, labels, group in clean:
                assert labels == source_labels[rid] and group not in groups
                groups.add(group)


def test_official_snapshot_preserves_source_text_labels_and_order():
    import csv
    from echolens.experiments import ROOT, configuration
    config = json.loads((ROOT / "configs/asap.json").read_text(encoding="utf-8"))
    folder = ROOT / "data/processed/stage1"
    if not (folder / "manifest.json").exists():
        pytest.skip("Prepare full dataset first")
    _, aspects = configuration()
    with duckdb.connect() as con:
        for split in ("train", "dev", "test"):
            source = ROOT / "data/raw/asap" / config["revision"] / f"data/{split}.csv"
            with source.open(encoding="utf-8-sig", newline="") as stream:
                original = {row["id"]: row for row in csv.DictReader(stream)}
            records = con.execute("SELECT source_record_id, text_raw, text_normalized, labels, created_at FROM read_parquet(?)", [str(folder / f"official-{split}.parquet")]).fetchall()
            assert len(records) == len(original)
            for sid, raw, normalized, labels, created_at in records:
                row = original[sid]
                assert raw == row["review"] and normalized == normalize(raw)
                assert labels == [int(row[a]) for a in aspects]
                assert created_at is None
