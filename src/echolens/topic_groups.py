"""Discover semantic subgroups within predicted aspects, keeping sparse groups out."""
from collections import defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def group_opinions(units, embeddings, distance_threshold=.30, min_feedbacks=3, linkage='complete'):
    groups = defaultdict(list)
    for i, unit in enumerate(units):
        if unit["aspect"] != "unknown" and unit.get("scope","current") == "current":
            groups[unit["aspect"]].append(i)
    labels = np.full(len(units), -1, dtype=int)
    next_label = 0
    for aspect in sorted(groups):
        indices = np.array(groups[aspect])
        if len({units[i]["record_id"] for i in indices}) < min_feedbacks:
            continue
        local = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage=linkage,
                                       distance_threshold=distance_threshold).fit_predict(embeddings[indices])
        for label in sorted(set(local)):
            members = indices[local == label]
            if len({units[i]["record_id"] for i in members}) >= min_feedbacks:
                labels[members] = next_label
                next_label += 1
    return labels


def representative_units(units, count=3, boundary=False):
    """One fragment per record; boundary is within-topic distance, not a negative label."""
    chosen, seen = [], set()
    for unit in sorted(units, key=lambda u:u["centrality"] if boundary else -u["centrality"]):
        if unit["record_id"] not in seen:
            chosen.append(unit)
            seen.add(unit["record_id"])
            if len(chosen) == count:
                break
    return chosen
