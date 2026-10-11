"""Third-iteration overlap deduplication; extraction trial archived locally."""


def deduplicate(opinions, text):
    """Collapse strongly overlapping same-aspect evidence, never opposite sentiments."""
    result = []
    for op in sorted(opinions, key=lambda x:(x['start'], -(x['end']-x['start']))):
        match = next((p for p in result if
            all(p[k] == op[k] for k in ('aspect','sentiment','intent','scope')) and
            (p['target'] == op['target'] or not p['target'] or not op['target']) and
            max(0,min(p['end'],op['end'])-max(p['start'],op['start'])) /
            min(p['end']-p['start'],op['end']-op['start']) >= .85), None)
        if match is None:
            result.append(dict(op))
            continue
        match['start'],match['end'] = min(match['start'],op['start']),max(match['end'],op['end'])
        match['sentence_start'],match['sentence_end'] = min(match['sentence_start'],op['sentence_start']),max(match['sentence_end'],op['sentence_end'])
        match['evidence'] = match['issue'] = text[match['start']:match['end']]
        match['target'] = match['target'] or op['target']
        match['raw_indices'] += op['raw_indices']
        match['review_flags'] = list(dict.fromkeys(match.get('review_flags',[])+op.get('review_flags',[])))
    return result
