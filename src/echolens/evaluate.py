"""ASAP metrics that do not reward the abundance of not-mentioned labels."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def multilabel_report(truth, prediction, names):
    p, r, f, s = precision_recall_fscore_support(truth, prediction, average=None, zero_division=0)
    micro = precision_recall_fscore_support(truth, prediction, average="micro", zero_division=0)
    return {"macro_f1": float(np.mean(f)), "micro_f1": float(micro[2]),
            "micro_precision": float(micro[0]), "micro_recall": float(micro[1]),
            "per_label": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, name in enumerate(names)}}


def evaluate(truth, prediction, aspects):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != len(aspects) or not len(truth):
        raise ValueError("Expected equal, nonempty [rows, aspects] arrays")
    if not np.isin(truth, [-2, -1, 0, 1]).all() or not np.isin(prediction, [-2, -1, 0, 1]).all():
        raise ValueError("Unknown original ASAP label")
    mention = multilabel_report(truth != -2, prediction != -2, aspects)
    classes = [-1, 0, 1]
    pairs = multilabel_report(np.stack([truth == c for c in classes], axis=2).reshape(len(truth), -1),
                              np.stack([prediction == c for c in classes], axis=2).reshape(len(truth), -1),
                              [f"{a}/{c}" for a in aspects for c in classes])
    mask = truth != -2
    t, y = truth[mask], prediction[mask]
    p, r, f, s = precision_recall_fscore_support(t, y, labels=classes, zero_division=0)
    sentiment = {"macro_f1": float(np.mean(f)), "accuracy": float(np.mean(t == y)) if len(t) else 0.0,
                 "predicted_mention_coverage": float(np.mean(y != -2)) if len(t) else 0.0, "positions": len(t),
                 "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes)}}
    return {"rows": len(truth), "aspects": mention, "pairs": pairs, "sentiment_on_gold_mentions": sentiment}
