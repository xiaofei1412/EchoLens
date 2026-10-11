"""Fit vocabulary on train only, select C on dev, never read test here."""
import argparse
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from echolens.evaluate import evaluate
from echolens.experiments import ROOT, configuration, load_split, provenance, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["official", "clean"], required=True)
    args = parser.parse_args()
    config, aspects = configuration()
    cfg = config["tfidf"]
    output = ROOT / "artifacts/stage-01" / f"tfidf-{args.version}"
    if (output / "selection.json").exists():
        raise FileExistsError("Completed run exists; preserve it before rerunning")
    output.mkdir(parents=True, exist_ok=True)
    ids, texts, truth = load_split(args.version, "train")
    dev_ids, dev_texts, dev_y = load_split(args.version, "dev")
    assert not set(ids) & set(dev_ids)
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["min_df"], max_features=cfg["max_features"], sublinear_tf=True, lowercase=False, dtype=np.float32)
    train_x = vectorizer.fit_transform(texts)
    dev_x = vectorizer.transform(dev_texts)
    selection, best = [], -1.0
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for c in cfg["C_candidates"]:
            classifiers = []
            for a, name in enumerate(aspects):
                classifier = LinearSVC(C=c, max_iter=cfg["max_iter"], random_state=config["seed"], dual="auto")
                classifier.fit(train_x, truth[:, a])
                classifiers.append(classifier)
                print(f"{args.version} C={c} aspect={name}", flush=True)
            predictions = np.column_stack([m.predict(dev_x) for m in classifiers])
            metrics = evaluate(dev_y, predictions, aspects)
            score = metrics["pairs"]["micro_f1"]
            selection.append({"C": c, "dev": metrics})
            if score > best:
                best = score
                joblib.dump({"vectorizer": vectorizer, "classifiers": classifiers, "aspects": aspects}, output / "model.joblib", compress=3)
                saved_prediction = predictions.copy()
                best_c = c
            print(f"C={c} dev pair micro_f1={score:.6f}", flush=True)
    model = joblib.load(output / "model.joblib")
    reload_x = model["vectorizer"].transform(dev_texts)
    reloaded = np.column_stack([m.predict(reload_x) for m in model["classifiers"]])
    assert np.array_equal(saved_prediction, reloaded)
    report = {"model": "char-tfidf-linearsvc", "data_version": args.version, "config": config, **provenance(),
              "selected_C": best_c, "candidates": selection, "train_rows": len(ids), "dev_rows": len(dev_ids),
              "features": train_x.shape[1], "seconds": time.perf_counter() - started,
              "reload_predictions_equal": True, "model_sha256": sha256(output / "model.joblib"), "test_read": False}
    save_json(output / "selection.json", report)
    save_json(ROOT / "docs/reports" / f"stage1-tfidf-{args.version}-dev.json", report)


if __name__ == "__main__":
    main()
