"""Convert the fixed 900 teacher records and freeze a grouped 810/90 split."""
import json
import random
from collections import defaultdict

from transformers import AutoTokenizer
from echolens.stage3_sft import ROOT, sha256, atomic_json, read_json, FIELDS, encode, parse_output, system_prompt

OUT = ROOT / "data/training/stage3-sft-v1"
SEED = 20261011


def main():
    if (OUT / "manifest.json").exists():
        m = read_json(OUT / "manifest.json")
        assert all(sha256(OUT / f) == h for f, h in m["files"].items())
        print("Existing frozen SFT package verified")
        return
    aspects = read_json(ROOT / "configs/labels/teacher-v3.json")["aspects"]
    model = read_json(ROOT / "configs/models.json")["generation"]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / model["local_dir"], local_files_only=True)
    prompt = system_prompt(aspects)
    parent = read_json(ROOT / "data/annotations/stage3-teacher-900-v2/manifest.json")
    groups = {rid: group for b in parent["batches"].values() for rid, group in zip(b["record_ids"], b["duplicate_group_ids"])}
    packages = []
    for name in ("stage3-ai-revised20-v3", "stage3-ai-revised80-v3.2"):
        p = ROOT / "data/annotations" / name
        m = read_json(p / "final-manifest.json")
        assert all(sha256(p / f) == h for f, h in m["files"].items())
        packages.append((p, m["final_outputs"]))
    p = ROOT / "data/annotations/stage3-teacher-wave100-v3.2"
    versions = read_json(ROOT / "docs/reports/stage3-wave100-complete-review.json")["final_versions"]
    packages.append((p, {b: f"outputs/{b}.{v}.json" for b, v in versions.items()}))
    p = ROOT / "data/annotations/stage3-teacher-final700-v3.2"
    packages.append((p, read_json(p / "reviews/completion-integrity-v1/final-selection.json")["final_outputs"]))
    rows, source_hashes = [], {}
    for p, files in packages:
        for batch, file in files.items():
            inp = p / "inputs" / f"{batch}.json"
            inputs = {r["record_id"]: r for r in read_json(inp)["records"]}
            output = p / file
            source_hashes[str(inp.relative_to(ROOT))] = sha256(inp)
            source_hashes[str(output.relative_to(ROOT))] = sha256(output)
            receipt = read_json(p / "checks" / output.name) if (p / "checks" / output.name).exists() else {}
            if receipt.get("response_sha256"):
                assert receipt["response_sha256"] == sha256(output)
            for r in read_json(output)["records"]:
                src = inputs[r["record_id"]]
                answer = json.dumps({"opinions": [{k: o[k] for k in FIELDS} for o in r["opinions"]]}, ensure_ascii=False, separators=(",", ":"))
                checked = parse_output(answer, src["text_raw"], aspects)
                assert checked["schema_valid"] and not checked["errors"]
                ids, labels = encode(tokenizer, prompt, src["text_raw"], answer)
                rows.append({"record_id": r["record_id"], "source_record_id": src["source_record_id"],
                             "group": groups[r["record_id"]], "text": src["text_raw"], "answer": answer,
                             "tokens": len(ids), "answer_tokens": sum(n != -100 for n in labels),
                             "source_package": p.name, "source_output": file,
                             "teacher": receipt.get("declared_model", "teacher_with_ai_revision"),
                             "coverage": r["coverage"], "review_questions": r["review_questions"]})
    assert len(rows) == len({r["record_id"] for r in rows}) == len({r["group"] for r in rows}) == 900
    assert {r["record_id"] for r in rows} == set(groups)
    # All duplicate groups are singletons; stratify by provenance and raw-text length.
    strata = defaultdict(list)
    for r in rows:
        strata[(r["teacher"], 0 if len(r["text"]) <= 200 else 1 if len(r["text"]) <= 500 else 2)].append(r)
    rng = random.Random(SEED)
    allocation = {k: len(v) // 10 for k, v in strata.items()}
    for k in sorted(strata, key=lambda k: (-(len(strata[k]) % 10), k))[:90-sum(allocation.values())]:
        allocation[k] += 1
    dev_ids = set()
    for k in sorted(strata):
        subset = sorted(strata[k], key=lambda r: r["record_id"])
        rng.shuffle(subset)
        dev_ids.update(r["record_id"] for r in subset[:allocation[k]])
    train, dev = [], []
    for r in rows:
        (dev if r["record_id"] in dev_ids else train).append(r)
    assert len(train) == 810 and len(dev) == 90
    assert not {r["group"] for r in train} & {r["group"] for r in dev}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("dev.jsonl", dev)):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    (OUT / "system.txt").write_text(prompt, encoding="utf-8")
    atomic_json(OUT / "aspects.json", aspects)
    lengths = sorted(r["tokens"] for r in rows)
    report = {"records": 900, "opinions": sum(len(json.loads(r["answer"])["opinions"]) for r in rows),
              "train": len(train), "dev": len(dev), "group_overlap": 0, "seed": SEED,
              "token_lengths": {"min": min(lengths), "median": lengths[450], "p95": lengths[854], "max": max(lengths)},
              "over_4096_train": sum(r["tokens"] > 4096 for r in train),
              "over_4096_dev": sum(r["tokens"] > 4096 for r in dev),
              "max_answer_tokens": max(r["answer_tokens"] for r in rows),
              "human_gold": False, "teacher_supervised_experiment": True, "semantic_edits": 0}
    atomic_json(OUT / "manifest.json", {**report, "model": model, "source_hashes": source_hashes,
                "files": {f: sha256(OUT / f) for f in ["train.jsonl", "dev.jsonl", "system.txt", "aspects.json"]}})
    atomic_json(ROOT / "docs/reports/stage3-sft-preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
