"""Verify completed frozen runs and publish only aggregate full-training results."""
import json
from echolens.stage3_sft import ROOT, read_json, sha256, atomic_json, metrics


def main():
    run = ROOT / 'artifacts/stage-03/sft-full-v2'
    base_dir = ROOT / 'artifacts/stage-03/sft-baseline-v1'
    eval_dir = ROOT / 'artifacts/stage-03/sft-full-v2-eval'
    train = read_json(run/'report.json')
    base, tuned = [read_json(p/'report.json') for p in [base_dir, eval_dir]]
    assert all(r['completed'] for r in [train, base, tuned])
    assert tuned['adapter_reload_equal']
    selected = read_json(run/'selected.json')
    assert selected['step'] == train['best_step']
    assert selected['dev_loss'] == min(r['dev_loss'] for r in train['evaluations'] if r['step'])
    data = ROOT/'data/training/stage3-sft-v1'
    m = read_json(data/'manifest.json')
    assert all(sha256(data/p) == h for p,h in m['files'].items())
    assert all(sha256(ROOT/p) == h for p,h in m['source_hashes'].items())
    for report in [train, base, tuned]:
        assert report['identity']['data_manifest_sha256'] == sha256(data/'manifest.json')
        assert report['identity']['model'] == m['model']
    for key in ['max_new_tokens', 'max_length', 'metrics_sha256']:
        assert base['identity'][key] == tuned['identity'][key]
    assert base['versions'] == tuned['versions'] and base['gpu'] == tuned['gpu']
    assert tuned['identity']['metrics_sha256'] == sha256(ROOT/'src/echolens/stage3_sft.py')
    assert tuned['identity']['adapter']['path'] == selected['adapter']
    for p,h in tuned['identity']['adapter']['files'].items():
        assert sha256(ROOT/selected['adapter']/p) == h
    dev = [json.loads(line) for line in (data/'dev.jsonl').read_text(encoding='utf-8').splitlines()]
    for folder, report in [(base_dir, base), (eval_dir, tuned)]:
        rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == 90 and {r['record_id'] for r in rows} == {r['record_id'] for r in dev}
        assert metrics(rows, dev, read_json(data/'aspects.json')) == report['metrics']
    for epoch in range(1, train['completed_epochs']+1):
        history = [r for r in train['train_history'] if r['epoch'] == epoch]
        assert len(history) == 51 and sum(r['records'] for r in history) == 810
    for checkpoint in {train['last_checkpoint'], train['best_checkpoint']}:
        sealed = read_json(run/checkpoint/'files.json')
        assert all(sha256(run/checkpoint/p) == h for p,h in sealed.items())
    comparison = {'reference': 'AI teacher agreement, not human gold', 'records': 90,
        'completed_epochs': train['completed_epochs'], 'best_step': train['best_step'],
        'dev_loss_evaluations': train['evaluations'], 'second_epoch_decision': train['second_epoch_decision'],
        'baseline_metrics': base['metrics'], 'tuned_metrics': tuned['metrics'],
        'delta': {k: tuned['metrics'][k]-base['metrics'][k] for k in ['json_valid_rate', 'schema_valid_rate', 'evidence_valid_rate']},
        'joint_f1_delta': {mode: tuned['metrics']['joint_evidence_aspect_sentiment'][mode]['f1']-
            base['metrics']['joint_evidence_aspect_sentiment'][mode]['f1'] for mode in ['strict', 'relaxed']},
        'baseline_mean_seconds': base['mean_seconds'], 'tuned_mean_seconds': tuned['mean_seconds'],
        'baseline_output_limit_records': base['output_limit_records'], 'tuned_output_limit_records': tuned['output_limit_records'],
        'verified_source_files': len(m['source_hashes']), 'same_ids_and_metrics_verified': True}
    dest = ROOT/'docs/reports'
    atomic_json(dest/'stage3-sft-full-training.json', train)
    atomic_json(dest/'stage3-sft-full-evaluation.json', tuned)
    atomic_json(dest/'stage3-sft-full-comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
