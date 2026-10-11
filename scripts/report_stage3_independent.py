"""Verify the frozen paired run and publish only mechanical aggregate results."""
import json
from collections import Counter

import duckdb
from echolens.independent_review import PACKET, RUN, load_packet, predictions, history
from echolens.stage3_sft import ROOT, read_json, sha256, atomic_json, parse_output
from prepare_stage3_holdout import collect_ids


def main():
    manifest, rows, aspects = load_packet()
    for group in ['source_files', 'snapshot_files']:
        assert all(sha256(ROOT/p)==h for p,h in manifest[group].items()), group
    training = ROOT/'data/training/stage3-sft-v1'
    assert sha256(training/'manifest.json') == manifest['training_manifest_sha256']
    original = read_json(training/'manifest.json')
    assert all(sha256(training/p)==h for p,h in original['files'].items())
    for name in ['system.txt','aspects.json']:
        assert sha256(PACKET/name)==sha256(training/name)
    assert all(sha256(ROOT/manifest['adapter']/p)==h for p,h in manifest['adapter_files'].items())

    exposed = set()
    for relative in manifest['source_files']:
        path = ROOT/relative
        if path.suffix=='.jsonl':
            for line in path.read_text(encoding='utf-8').splitlines():
                collect_ids(json.loads(line),exposed)
        else:
            collect_ids(read_json(path),exposed)
    with duckdb.connect() as db:
        data = db.execute('SELECT record_id,source_record_id,text_raw,text_normalized,duplicate_group_id,split FROM read_parquet(?)',
            [str(ROOT/'data/processed/stage1/official-*.parquet')]).fetchall()
    source = {r[0]:r for r in data}
    blocked = [r for r in data if r[5] in {'train','dev'} or r[0] in exposed or str(r[1]) in exposed]
    selected = [source[r['record_id']] for r in rows]
    assert len(rows)==len({r['record_id'] for r in rows})==50
    assert len({r[3] for r in selected})==len({r[4] for r in selected})==50
    assert all(r['text']==s[2] and r['source_record_id']==s[1] and s[5]=='test' for r,s in zip(rows,selected))
    overlaps = {name:len({r[col] for r in selected}&{r[col] for r in blocked})
                for name,col in [('record_ids',0),('source_ids',1),('normalized_texts',3),('duplicate_groups',4)]}
    assert not any(overlaps.values())
    sft_ids = {json.loads(line)['record_id'] for name in ['train.jsonl','dev.jsonl']
               for line in (training/name).read_text(encoding='utf-8').splitlines()}
    assert len(sft_ids)==900 and not sft_ids & {r['record_id'] for r in rows}

    reports, aggregates = {}, {}
    by_id = {r['record_id']:r for r in rows}
    for side,mode in [('before','baseline'),('after','evaluate')]:
        report = read_json(RUN/side/'report.json')
        assert report and report['completed'] and report['completed_records']==50, side+' not finished'
        preds = predictions(RUN/side)
        assert set(preds)==set(by_id) and len(preds)==50
        identity = report['identity']
        assert identity['max_new_tokens']==4096
        assert identity['mode']==mode and identity['data_manifest_sha256']==sha256(PACKET/'manifest.json')
        assert identity['model']==manifest['model']
        assert identity['script_sha256']==sha256(RUN/'source/run_stage3_sft.py')
        assert identity['metrics_sha256']==sha256(RUN/'source/stage3_sft.py')
        checks = [parse_output(p['output'],by_id[rid]['text'],aspects) for rid,p in preds.items()]
        for key in ['json_valid','schema_valid']:
            assert report['mechanical_checks'][key]==sum(c[key] for c in checks)
        assert 'metrics' not in report and not report['human_gold']
        reports[side] = report
        aggregates[side] = {key:report[key] for key in ['mechanical_checks','mean_seconds','median_seconds','output_limit_records','peak_allocated_gib']}
        aggregates[side].update(generation_seconds=sum(p['seconds'] for p in preds.values()),
            predictions_sha256=sha256(RUN/side/'predictions.jsonl'),report_sha256=sha256(RUN/side/'report.json'))
    for key in ['model','data_manifest_sha256','max_new_tokens','script_sha256','metrics_sha256']:
        assert reports['before']['identity'][key]==reports['after']['identity'][key]
    assert reports['before']['versions']==reports['after']['versions']
    assert reports['after']['identity']['adapter']['files']==manifest['adapter_files']
    assert reports['after']['adapter_reload_equal']
    reviews = history()
    summary = {'records':50,'predictions':100,'source_overlaps':overlaps,'sft_900_overlap':0,
        'length_strata':dict(Counter(r['length_band'] for r in rows)),
        'manifest_sha256':sha256(PACKET/'manifest.json'),'model':manifest['model'],
        'adapter':manifest['adapter'],'versions':reports['before']['versions'],
        'generation':{'do_sample':False,'thinking':False,'max_new_tokens':4096,'quantization':'NF4 double quant'},
        'runs':aggregates,'human_review_events_at_verification':len(reviews),'human_gold':False,
        'limitations':manifest['limitations'],'interpretation':'Mechanical checks only; no semantic accuracy or human F1'}
    atomic_json(ROOT/'docs/reports/stage3-independent50-inference.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
