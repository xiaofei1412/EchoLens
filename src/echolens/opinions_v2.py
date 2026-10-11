"""Context-aware opinion contract, separate from the unchanged ASAP classifier."""
import re


def clauses(text):
    """Split noisy clause boundaries while keeping raw offsets, including literal \\n."""
    result, start = [], 0
    for match in re.finditer(r"[，,。！？!?；;\n]+|\\+n|$", text):
        end = match.end()
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right-1].isspace():
            right -= 1
        if left < right:
            result.append({"id":len(result),"start":left,"end":right,"text":text[left:right]})
        start = end
    return result


def merge_adjacent(opinions, text):
    """Merge nearby repetitions only when target, aspect, sentiment AND intent agree."""
    merged = []
    for opinion in sorted(opinions, key=lambda o:(o["sentence_start"], o["sentence_end"])):
        previous = next((p for p in reversed(merged) if
            all(p[k] == opinion[k] for k in ("target", "aspect", "sentiment", "intent")) and
            p.get("scope","current") == opinion.get("scope","current") and
            (opinion["target"] or (p["start"],p["end"]) == (opinion["start"],opinion["end"])) and
            opinion["sentence_start"] <= p["sentence_end"] + (1 if opinion["target"] else 0)), None)
        if previous is None:
            merged.append(dict(opinion))
        else:
            quoted_issues = previous["issue"] == previous["evidence"] and opinion["issue"] == opinion["evidence"]
            previous["start"] = min(previous["start"], opinion["start"])
            previous["sentence_start"] = min(previous["sentence_start"], opinion["sentence_start"])
            previous["end"] = max(previous["end"], opinion["end"])
            previous["sentence_end"] = max(previous["sentence_end"], opinion["sentence_end"])
            previous["evidence"] = text[previous["start"]:previous["end"]]
            if quoted_issues:
                previous["issue"] = previous["evidence"]
            elif opinion["issue"] != previous["issue"]:
                previous["issue"] += "；" + opinion["issue"]
            previous["raw_indices"] = previous["raw_indices"] + opinion["raw_indices"]
            if "review_flags" in opinion:
                previous["review_flags"] = list(dict.fromkeys(previous.get("review_flags",[])+opinion["review_flags"]))
    return merged
