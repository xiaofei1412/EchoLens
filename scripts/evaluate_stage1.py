"""Freeze all four selected models before opening test; publish aggregate metrics."""
import json
import time
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer
from echolens.encoder import AspectEncoder, tokenized_loader, predict
from echolens.evaluate import evaluate
from echolens.experiments import ROOT, configuration, load_split, provenance, save_json, sha256


def main():
    config, aspects = configuration()
    folder = ROOT / "artifacts/stage-01"
    runs = [(kind, version) for kind in ("tfidf", "encoder") for version in ("official", "clean")]
    frozen = {"provenance": provenance(), "models": {}}
    for kind, version in runs:
        path = folder / f"{kind}-{version}"
        selection = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        assert selection["config"] == config and selection["snapshot_sha256"] == frozen["provenance"]["snapshot_sha256"]
        weights = path / ("model.joblib" if kind == "tfidf" else "best.pt")
        assert sha256(weights) == selection["model_sha256"], "Weights changed after dev selection"
        frozen["models"][path.name] = {"model_sha256": selection["model_sha256"], "selection_sha256": sha256(path / "selection.json")}
    freeze_path = folder / "test-freeze.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text(encoding="utf-8")) == frozen, "Cannot change a frozen test experiment"
    else:
        save_json(freeze_path, frozen)
    # All choices and model hashes frozen above. From here on, no fitting or selection.
    torch.set_num_threads(4)
    for kind, version in runs:
        path = folder / f"{kind}-{version}"
        if (path / "test.json").exists():
            print(f"Preserving completed test report: {path.name}", flush=True)
            continue
        if kind == "tfidf":
            model = joblib.load(path / "model.joblib")
        else:
            base = ROOT / config["encoder"]["local_dir"]
            tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
            model = AspectEncoder(base, len(aspects), pretrained=False).to("cuda")
            model.load_state_dict(torch.load(path / "best.pt", map_location="cpu", weights_only=True))
        results = {}
        # Both test views allow a common-denominator comparison, without selecting on either.
        for test_view in ("official", "clean"):
            ids, texts, truth = load_split(test_view, "test")
            if kind == "tfidf":
                started = time.perf_counter()
                x = model["vectorizer"].transform(texts)
                prediction = np.column_stack([m.predict(x) for m in model["classifiers"]])
                timing = {"seconds": time.perf_counter() - started, "scope": "TF-IDF transform + all 18 classifier predictions; excludes loading"}
            else:
                loader = tokenized_loader(tokenizer, texts, truth, config["encoder"])
                prediction, timing = predict(model, loader)
                lengths = [len(x) for x in tokenizer(texts, truncation=False, verbose=False)["input_ids"]]
                timing["truncated_rows"] = sum(n > config["encoder"]["max_length"] for n in lengths)
            np.savez_compressed(path / f"test-{test_view}-predictions.npz", ids=np.asarray(ids), truth=truth, prediction=prediction)
            results[test_view] = {"metrics": evaluate(truth, prediction, aspects), "timing": timing}
            print(f"{path.name} test={test_view} pair micro_f1={results[test_view]['metrics']['pairs']['micro_f1']:.6f}", flush=True)
        report = {"run": path.name, "test_freeze_sha256": sha256(freeze_path), "results": results,
                  "limitations": ["Official training view contains known cross-split duplicate groups", "Single seed; no significance or cross-domain claim", "Pretrained encoder may have seen public data; this audit only addresses supervised split duplicates"]}
        save_json(path / "test.json", report)
        save_json(ROOT / "docs/reports" / f"stage1-{path.name}-test.json", report)
        del model
        torch.cuda.empty_cache()
    save_json(ROOT / "docs/reports/stage1-test-freeze.json", frozen)


if __name__ == "__main__":
    main()
