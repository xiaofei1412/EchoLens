"""Train the frozen 810 records for one, conditionally two, resumable epochs."""
import argparse
import importlib.metadata
import json
import random
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, set_seed

from echolens.stage3_sft import ROOT, sha256, atomic_json, read_json, encode
from echolens.stage3_training import epoch_batches, second_epoch_decision
from probe_model import load_model

DATA = ROOT / 'data/training/stage3-sft-v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(DATA / 'manifest.json')
    assert all(sha256(DATA / p) == h for p, h in manifest['files'].items())
    baseline = read_json(ROOT / 'artifacts/stage-03/sft-baseline-v1/report.json')
    assert baseline['completed'] and baseline['identity']['data_manifest_sha256'] == sha256(DATA / 'manifest.json')
    model_config = manifest['model']
    identity = {'data_manifest_sha256': sha256(DATA / 'manifest.json'), 'model': model_config,
                'code': {p: sha256(ROOT / p) for p in ['scripts/train_stage3_full.py',
                    'src/echolens/stage3_training.py', 'src/echolens/stage3_sft.py', 'scripts/probe_model.py']},
                'seed': 20261011, 'rank': 8, 'alpha': 16, 'dropout': 0.05, 'target_modules': 'all-linear text only',
                'learning_rate': 1e-4, 'warmup_steps': 3, 'schedule': 'constant after warmup',
                'optimizer': 'AdamW', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.01,
                'gradient_clip': 1.0, 'micro_batch': 1, 'gradient_accumulation': 16,
                'partial_batch': 'divide loss by actual group size', 'max_length': 4096,
                'quantization': 'NF4 double quant BF16 compute', 'autocast': 'bfloat16',
                'memory_policy': 'empty CUDA cache after every variable-length train/eval sample',
                'supervision': 'assistant only', 'gradient_checkpointing': 'non-reentrant',
                'eval_steps': [0, 25, 51, 76, 102], 'max_epochs': 2,
                'second_epoch_rule': 'epoch1 endpoint below initial and at least 1% below step25',
                'selection': 'minimum mean per-record dev loss among trained checkpoints'}
    prior = read_json(out / 'report.json')
    if prior:
        assert prior['identity'] == identity, 'Code/config changed; use a new output directory'
        if prior.get('completed'):
            print('Completed training preserved', flush=True)
            return
    report = prior or {'identity': identity, 'started_at': datetime.now(timezone.utc).isoformat(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate']},
        'gpu': torch.cuda.get_device_name(), 'human_gold': False, 'completed': False,
        'train_records': 810, 'dev_records': 90, 'train_history': [], 'evaluations': [],
        'training_seconds': 0.0, 'eval_seconds': 0.0, 'peak_allocated_gib': 0.0}
    def save():
        atomic_json(out / 'report.json', report)
    save()
    source = out / 'source'
    source.mkdir(exist_ok=True)
    for p in identity['code']:
        (source / p.replace('/', '__')).write_bytes((ROOT / p).read_bytes())
    set_seed(identity['seed'])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / model_config['local_dir'], local_files_only=True)
    prompt = (DATA / 'system.txt').read_text(encoding='utf-8')
    rows = {split: [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            for split in ['train', 'dev']}
    assert len(rows['train']) == 810 and len(rows['dev']) == 90
    encoded = {split: [encode(tokenizer, prompt, r['text'], r['answer']) for r in records]
               for split, records in rows.items()}
    assert all(len(ids) <= 4096 for pairs in encoded.values() for ids, _ in pairs)
    def batch(pair):
        ids, labels = pair
        return {'input_ids': torch.tensor([ids], device='cuda'), 'labels': torch.tensor([labels], device='cuda'),
                'attention_mask': torch.ones((1, len(ids)), dtype=torch.long, device='cuda')}
    print('Loading fixed base; new adapter unless resuming own checkpoint', flush=True)
    model = prepare_model_for_kbit_training(load_model(ROOT / model_config['local_dir']),
        use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    latest = read_json(out / 'latest.json')
    state = None
    if latest:
        checkpoint = out / latest['checkpoint']
        sealed = read_json(checkpoint / 'files.json')
        assert all(sha256(checkpoint / p) == h for p, h in sealed.items())
        state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
        assert state['report']['identity'] == identity
        model = PeftModel.from_pretrained(model, checkpoint / 'adapter', is_trainable=True)
        report = state['report']
        report.pop('error', None)
        report['resumed_from'] = latest['checkpoint']
    else:
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules='all-linear', task_type='CAUSAL_LM'))
        # Without a committed checkpoint any uncommitted progress is rerun from the fixed seed.
        report.update(train_history=[], evaluations=[], training_seconds=0.0, eval_seconds=0.0)
        report.pop('error', None)
        for key in ['best_checkpoint', 'best_dev_loss', 'best_step', 'last_checkpoint',
                    'completed_steps', 'completed_epochs', 'second_epoch_decision']:
            report.pop(key, None)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    start_epoch, start_batch = 0, 0
    if state:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        start_epoch, start_batch = state['next_epoch'], state['next_batch']
        del state
    torch.cuda.reset_peak_memory_stats()
    def evaluate(step, epoch):
        model.eval()
        started = time.perf_counter()
        losses = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for pair in encoded['dev']:
                loss = model(**batch(pair)).loss
                assert torch.isfinite(loss), 'Nonfinite validation loss'
                losses.append(float(loss))
                del loss
                torch.cuda.empty_cache()
        result = {'step': step, 'epoch': epoch, 'dev_loss': sum(losses)/len(losses),
                  'records': len(losses), 'seconds': time.perf_counter()-started}
        report['evaluations'].append(result)
        report['eval_seconds'] += result['seconds']
        atomic_json(out / f'dev-loss-{step:04d}.json', {'record_ids': [r['record_id'] for r in rows['dev']], 'losses': losses})
        print(f"eval step={step} dev_loss={result['dev_loss']:.6f} seconds={result['seconds']:.1f}", flush=True)
        model.train()
        return result
    def commit(step, next_epoch, next_batch):
        name = f'checkpoint-{step:04d}-{uuid.uuid4().hex[:6]}'
        dest = out / name
        dest.mkdir()
        model.save_pretrained(dest / 'adapter')
        tokenizer.save_pretrained(dest / 'adapter')
        report['last_checkpoint'] = name
        current = report['evaluations'][-1]
        if not report.get('best_checkpoint') or current['dev_loss'] < report['best_dev_loss']:
            report.update(best_checkpoint=name, best_dev_loss=current['dev_loss'], best_step=step)
        torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'next_epoch': next_epoch, 'next_batch': next_batch,
                    'report': report}, dest / 'state.pt')
        atomic_json(dest / 'files.json', {str(p.relative_to(dest)): sha256(p) for p in dest.rglob('*') if p.is_file()})
        atomic_json(out / 'latest.json', {'checkpoint': name, 'step': step})
        save()
    try:
        if not report['evaluations']:
            evaluate(0, 0)
            save()
        for epoch in range(start_epoch, 2):
            if epoch == 1 and not report.get('second_epoch_decision', {}).get('continue'):
                break
            groups = epoch_batches(810, epoch)
            model.train()
            for index in range(start_batch if epoch == start_epoch else 0, len(groups)):
                started = time.perf_counter()
                group = groups[index]
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for i in group:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = model(**batch(encoded['train'][i])).loss
                    assert torch.isfinite(loss), 'Nonfinite training loss'
                    total += float(loss.detach())
                    (loss/len(group)).backward()
                    del loss
                    torch.cuda.empty_cache()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                step = epoch*len(groups)+index+1
                optimizer.param_groups[0]['lr'] = 1e-4 * min(1.0, step/3)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                report['training_seconds'] += elapsed
                report['peak_allocated_gib'] = max(report['peak_allocated_gib'], torch.cuda.max_memory_allocated()/2**30)
                report['train_history'].append({'step': step, 'epoch': epoch+1, 'records': len(group),
                    'loss': total/len(group), 'gradient_norm': float(norm), 'seconds': elapsed,
                    'learning_rate': optimizer.param_groups[0]['lr']})
                report['completed_steps'] = step
                save()
                print(f"train step={step} epoch={epoch+1} batch={index+1}/51 loss={total/len(group):.6f} seconds={elapsed:.1f} peak={report['peak_allocated_gib']:.2f}GiB", flush=True)
                if index == 24 or index == len(groups)-1:
                    result = evaluate(step, epoch+1)
                    if index == len(groups)-1:
                        report['completed_epochs'] = epoch+1
                        if epoch == 0:
                            ev = {r['step']: r['dev_loss'] for r in report['evaluations']}
                            report['second_epoch_decision'] = second_epoch_decision(ev[0], ev[25], ev[51])
                            print('Second epoch decision: '+json.dumps(report['second_epoch_decision']), flush=True)
                        commit(step, epoch+1, 0)
                    else:
                        commit(step, epoch, index+1)
        report.update(completed=True, finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'selected.json', {'checkpoint': report['best_checkpoint'], 'step': report['best_step'],
            'dev_loss': report['best_dev_loss'], 'adapter': str((out/report['best_checkpoint']/'adapter').relative_to(ROOT))})
        save()
        print('Training complete; selected '+report['best_checkpoint'], flush=True)
    except Exception as exc:
        report['error'] = repr(exc)
        report['peak_allocated_gib'] = max(report['peak_allocated_gib'], torch.cuda.max_memory_allocated()/2**30)
        save()
        raise


if __name__ == '__main__':
    main()
