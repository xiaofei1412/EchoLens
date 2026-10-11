"""Preserve prior runs; prepare a third-iteration train-only development snapshot."""
import random
import shutil
from echolens.experiments import ROOT, sha256
from echolens.workbench import dataset_path, records_at, read_json, save_dataset, atomic_json


def main():
    source, target = dataset_path('asap-demo-1000-v2'), dataset_path('asap-demo-1000-v3')
    if (target/'dataset.json').exists():
        print('Existing v3 preserved')
        return
    rows = records_at(source)
    save_dataset(target,rows,{'name':'ASAP · 1,000 条 · 第三轮修订 v3','kind':'train_demo',
        'analysis_config':'configs/stage2-v3.json','baseline_dataset':source.name})
    shutil.copy2(source/'classifications.json',target/'classifications.json')
    cases = {'5365','38689','6479','44572','32524','22304','26480','44206','33803','15748'}
    ids = [r['record_id'] for r in rows if r['source_record_id'] in cases]
    previous = read_json(source/'audit-selection.json')
    excluded = set(previous['pilot']+previous['additional_audit']+ids)
    remaining = [r['record_id'] for r in rows if r['record_id'] not in excluded]
    random.Random(20261011).shuffle(remaining)
    atomic_json(target/'audit-selection.json',{'purpose':'Train-only development, not human gold or held-out accuracy',
        'user_reported':ids,'pilot':ids+remaining[:22],'additional_audit':remaining[22:54]})
    atomic_json(target/'baseline-preservation.json',{name:{p.name:sha256(p) for p in
        [dataset_path(name)/'records.jsonl',dataset_path(name)/'classifications.json',dataset_path(name)/'topics.json']}
        for name in ('asap-demo-1000','asap-demo-1000-v2')})
    print('Prepared identical 1000 records, 32 development and 32 fresh audit records')


if __name__ == '__main__':
    main()
