"""Conservative lexical duplicate candidates with exact Jaccard verification."""
import re
from collections import defaultdict

from datasketch import MinHash, MinHashLSH


def duplicate_groups(records, config, seed):
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[root(j)] = root(i)

    lsh = MinHashLSH(threshold=config["lsh_threshold"], num_perm=config["num_perm"])
    exact, shingles, pairs = {}, {}, []
    k = config["char_ngram"]
    for i, row in enumerate(records):
        text = re.sub(r"\s+", "", row["text_raw"])
        if text in exact:
            j = exact[text]
            union(i, j)
            pairs.append({"left": records[j]["record_id"], "right": row["record_id"], "kind": "whitespace_exact", "jaccard": 1.0})
            continue
        exact[text] = i
        if len(text) < config["min_chars"]:
            continue
        grams = {text[n:n + k] for n in range(len(text) - k + 1)}
        mh = MinHash(num_perm=config["num_perm"], seed=seed)
        mh.update_batch([s.encode("utf-8") for s in sorted(grams)])
        for j in sorted(lsh.query(mh)):
            other = shingles[j]
            similarity = len(grams & other) / len(grams | other)
            if similarity >= config["verified_jaccard"]:
                union(i, j)
                pairs.append({"left": records[j]["record_id"], "right": row["record_id"], "kind": "char_ngram_jaccard", "jaccard": similarity})
        lsh.insert(i, mh)
        shingles[i] = grams
        if (i + 1) % 5000 == 0:
            print(f"Duplicate scan {i+1}/{len(records)}", flush=True)
    members = defaultdict(list)
    for i in range(len(records)):
        members[root(i)].append(i)
    groups = {}
    for indices in members.values():
        group_id = min(records[i]["record_id"] for i in indices)
        for i in indices:
            groups[records[i]["record_id"]] = group_id
    return groups, pairs


def clean_membership(records, groups):
    """Keep one deterministic record per group, with test > dev > train priority."""
    priority = {"test": 0, "dev": 1, "train": 2}
    kept = {}
    for row in sorted(records, key=lambda r: (priority[r["split"]], r["source_record_id"])):
        kept.setdefault(groups[row["record_id"]], row["record_id"])
    return set(kept.values())
