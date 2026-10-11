"""Verify frozen evidence and small integration runs; publish aggregates only."""
from collections import Counter

from echolens.candidate import validate_assets, digest, cached_result
from echolens.independent_review import PACKET, RUN, history
from echolens.stage3_sft import ROOT, sha256
from echolens.workbench import (dataset_path, records_at, results_at, read_json,
    atomic_json, report_markdown, review_reasons)


def main():
    expected = {
        PACKET/'manifest.json':'380c7f903e8a13f31d7b0bad2b4f67f670e36458833b4dcff4988122cd2ac574',
        RUN/'before/predictions.jsonl':'e0dd7af68cb46a8b672107ecf9179568929cd0fbf37f80a2f83b2147e5e8995e',
        RUN/'after/predictions.jsonl':'b147c622911712183ad92c5e57839c79485b1e5f513cf3374e37b66af793e657',
    }
    for path,value in expected.items():
        assert sha256(path)==value, str(path)
    config = validate_assets()
    acceptance = read_json(PACKET/'acceptance.json')
    assert acceptance['accepted'] and acceptance['packet_sha256']==sha256(PACKET/'manifest.json')
    saved = {e['record_id']:e for e in history(PACKET)}
    forms = [e for e in saved.values() if e['status']=='completed']
    folder = dataset_path('stage3-integration3-qlora-v1')
    rows = records_at(folder)
    classifications = read_json(folder/'classifications.json')
    extractions = results_at(folder)
    assert len(rows)==len(extractions)==len(classifications)==3
    assert read_json(folder/'job.json')['status']=='complete'
    assert not (ROOT/'artifacts/stage-02/gpu-job.lock').exists()
    identity = read_json(folder/'candidate-run.json')
    for row in rows:
        assert cached_result(folder/'extractions'/f"{row['record_id']}.json",digest(identity),row['text_raw'])
    previous = read_json(ROOT/'artifacts/stage-03/closeout/resume-before.json')
    path = ROOT/previous['path']
    assert sha256(path)==previous['sha256'] and path.stat().st_mtime_ns==previous['mtime_ns']
    topics = read_json(folder/'topics.json')
    assert all(extractions[u['record_id']]['valid'] for u in topics['units'])
    local = ROOT/'artifacts/stage-03/closeout'
    (local/'integration-report.md').write_text(report_markdown(folder,rows,classifications,extractions),encoding='utf-8')
    overflow = dataset_path('stage3-overflow-probe-qlora-v1')
    overflow_rows = records_at(overflow)
    failed = results_at(overflow)
    value = next(iter(failed.values()))
    assert not value['valid'] and not value['opinions'] and value['tokens']==0
    assert '输入超过4096' in value['error']
    assert review_reasons(value['record_id'],{},failed)
    failure_report = report_markdown(overflow,overflow_rows,{},failed)
    assert '原文 [' not in failure_report and '输入超过4096' in failure_report
    (local/'overflow-report.md').write_text(failure_report,encoding='utf-8')
    tests = (local/'tests.log').read_text(encoding='utf-8-sig')
    assert '62 passed' in tests
    report = {'stage':3,'status':'closed_experimental_scope','date':'2026-10-11',
        'candidate_config':config,'candidate_config_sha256':sha256(ROOT/'configs/stage3-candidate.json'),
        'frozen_holdout_hashes':{str(p.relative_to(ROOT)):v for p,v in expected.items()},
        'user_acceptance':{'source':'human_chat','batch_size':50,'accepted':True,
            'saved_completed_forms':len(forms),'saved_form_choices':dict(Counter(e['choice'] for e in forms)),
            'human_gold':False,'training_ready':False,'no_inferred_per_record_reviews':True},
        'integration':{'records':3,'valid_records':sum(v['valid'] for v in extractions.values()),
            'opinions':len(topics['units']),'topics':len(topics['topics']),
            'generation_seconds':[v['seconds'] for v in extractions.values()],
            'resume_first_file_bytes_and_mtime_unchanged':True,'job_complete':True,
            'scope':'train-pool engineering probe; not quality evaluation'},
        'failure_probe':{'synthetic_overflow_rejected':True,'model_generation_tokens':0,
            'excluded_from_evidence_export':True},
        'tests':{'passed':62,'log_sha256':sha256(local/'tests.log')},
        'limitations':['No full human gold F1; 900 AI labels instead of proposed 1500-3000 human-verified labels',
            'Independent50 candidate field validity 48/50 (96%), below 99% release goal',
            'Candidate remains optional; restaurant-only quality evidence, cross-domain unverified',
            'Resume tested at completed-record boundary, not mid-kernel numerical equivalence',
            'OOM not forcibly induced; exceptions stop worker and preserve completed records',
            'Three-record probe cannot validate topic naming or new full-dataset quality']}
    atomic_json(ROOT/'docs/reports/stage3-closeout.json',report)
    print('Verified: frozen holdout and adapter; 3-record resume; overflow quarantine; 62 tests.')


if __name__=='__main__':
    main()
