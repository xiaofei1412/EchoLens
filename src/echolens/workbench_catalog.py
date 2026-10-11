"""User-facing names and visibility; never mutate historical dataset metadata."""

ENGINEERING_KINDS = {'integration_probe', 'synthetic_fixture'}


def visible_datasets(items, include_history=False):
    if include_history:
        return items
    ids = {m['dataset_id'] for m in items}
    superseded = {'asap-demo-1000','asap-demo-1000-v2'} if 'asap-demo-1000-v3' in ids else set()
    return [m for m in items if m.get('kind') not in ENGINEERING_KINDS and m['dataset_id'] not in superseded]


def dataset_label(metadata):
    if metadata['dataset_id']=='asap-demo-1000-v3':
        return 'ASAP餐饮评价 · 1,000条基线演示'
    return metadata['name']


def preferred_dataset(items):
    candidates = [m for m in items if m.get('analysis_config')=='configs/stage3-candidate.json' and m.get('kind') not in ENGINEERING_KINDS]
    chosen = candidates or [m for m in items if m['dataset_id']=='asap-demo-1000-v3'] or items
    return chosen[0]['dataset_id'] if chosen else None
