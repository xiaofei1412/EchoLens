"""Freeze 50 test-pool comments, excluding all train/dev and prior development exposure."""
import json
import random

import duckdb
from echolens.stage3_sft import ROOT, read_json, sha256, atomic_json

OUT = ROOT / 'data/evaluation/stage3-independent50-v1'
SEED = 2026101201


def collect_ids(value, ids):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in {'record_id', 'source_record_id', 'source_id'} and isinstance(v, (str, int)):
                ids.add(str(v))
            elif k in {'record_ids', 'source_ids', 'excluded_source_ids'} and isinstance(v, list):
                ids.update(str(x) for x in v)
            else:
                collect_ids(v, ids)
    elif isinstance(value, list):
        for item in value:
            collect_ids(item, ids)


def main():
    if (OUT/'manifest.json').exists():
        m = read_json(OUT/'manifest.json')
        assert all(sha256(OUT/p) == h for p, h in m['files'].items())
        print('Frozen independent50 preserved; packet hashes verified')
        return
    prepared = ROOT/'data/processed/stage1'
    with duckdb.connect() as db:
        # Never select the upstream annotation/label columns.
        cursor = db.execute('SELECT record_id,source_record_id,text_raw,text_normalized,duplicate_group_id,split,clean_keep,rights_status FROM read_parquet(?) ORDER BY record_id',
                            [str(prepared/'official-*.parquet')])
        names = [c[0] for c in cursor.description]
        all_rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    exposure_files = set((ROOT/'data/exploration').glob('*.jsonl'))
    exposure_files.update((ROOT/'data/workbench').glob('*/records.jsonl'))
    for name in ['inputs.json', 'packet.json', 'manifest.json']:
        exposure_files.update((ROOT/'data/annotations').glob('*/'+name))
    exposure_files.update((ROOT/'data/annotations').glob('*/inputs/*.json'))
    ids = set()
    for p in sorted(exposure_files):
        if p.suffix == '.jsonl':
            for line in p.read_text(encoding='utf-8').splitlines():
                collect_ids(json.loads(line), ids)
        else:
            collect_ids(read_json(p), ids)
    # Include every official train/dev row, including excluded duplicates used by the official baselines.
    blocked = [r for r in all_rows if r['split'] in {'train', 'dev'} or r['record_id'] in ids or str(r['source_record_id']) in ids]
    blocked_ids = {str(r[k]) for r in blocked for k in ['record_id', 'source_record_id']}
    blocked_groups = {r['duplicate_group_id'] for r in blocked}
    blocked_texts = {r['text_normalized'] for r in blocked}
    candidates = [r for r in all_rows if r['split'] == 'test' and r['clean_keep']
        and r['rights_status'] == 'upstream_apache_2.0' and r['record_id'] not in blocked_ids
        and str(r['source_record_id']) not in blocked_ids and r['duplicate_group_id'] not in blocked_groups
        and r['text_normalized'] not in blocked_texts]
    rng = random.Random(SEED)
    initial, remaining = [], []
    for band, low, high, count, first in [('short',0,200,15,3),('medium',200,500,20,4),('long',500,100000,15,3)]:
        pool = [r for r in candidates if low < len(r['text_raw']) <= high]
        rng.shuffle(pool)
        assert len(pool) >= count
        chosen = [{**r, 'length_band': band} for r in pool[:count]]
        initial.extend(chosen[:first])
        remaining.extend(chosen[first:])
    rng.shuffle(initial)
    rng.shuffle(remaining)
    selected = initial + remaining
    assert len({r['record_id'] for r in selected}) == len({r['duplicate_group_id'] for r in selected}) == 50
    assert len({r['text_normalized'] for r in selected}) == 50
    packet = [{'record_id': r['record_id'], 'source_record_id': r['source_record_id'],
        'text': r['text_raw'], 'duplicate_group_id': r['duplicate_group_id'], 'source_split': 'test',
        'length_band': r['length_band'], 'review_order': i+1} for i,r in enumerate(selected)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in packet),encoding='utf-8')
    training = ROOT/'data/training/stage3-sft-v1'
    original = read_json(training/'manifest.json')
    assert all(sha256(training/p) == h for p,h in original['files'].items())
    for name in ['system.txt', 'aspects.json']:
        (OUT/name).write_bytes((training/name).read_bytes())
    selected_model = read_json(ROOT/'artifacts/stage-03/sft-full-v2/selected.json')
    adapter = ROOT/selected_model['adapter']
    summary = {'records':50, 'seed':SEED, 'length_strata':{'short':15,'medium':20,'long':15},
        'first10_strata':{'short':3,'medium':4,'long':3}, 'source_split':'test', 'eligible_records':len(candidates),
        'blocked_records':len(blocked), 'overlap_record_ids':0, 'overlap_duplicate_groups':0, 'overlap_normalized_texts':0,
        'human_reviews':0, 'human_gold':False, 'training_ready':False, 'model':original['model'],
        'usage':'independent paired human review; never train, tune prompts or select checkpoints on this packet',
        'limitations':['ASAP test was used for frozen stage1 classification evaluation, not training/tuning',
                      'Project-level exclusion; cannot guarantee unseen in foundation-model pretraining',
                      'Existing MinHash groups may miss semantic paraphrases; restaurant domain only',
                      'Human pairwise review does not produce full gold opinion spans or human F1']}
    manifest = {**summary, 'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(exposure_files)},
        'snapshot_files':{str(p.relative_to(ROOT)):sha256(p) for p in prepared.glob('official-*.parquet')},
        'training_manifest_sha256':sha256(training/'manifest.json'), 'adapter':selected_model['adapter'],
        'adapter_files':{n:sha256(adapter/n) for n in ['adapter_config.json','adapter_model.safetensors']},
        'files':{n:sha256(OUT/n) for n in ['records.jsonl','system.txt','aspects.json']}}
    atomic_json(OUT/'manifest.json',manifest)
    atomic_json(ROOT/'docs/reports/stage3-independent50-preparation.json',{**summary,'manifest_sha256':sha256(OUT/'manifest.json')})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
