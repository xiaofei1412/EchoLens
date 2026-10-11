"""Candidate GPU worker, run with .venv; no workbench/duckdb dependency."""
import argparse
import importlib.metadata
import json
import os
import re
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed

from probe_model import load_model
from echolens.stage3_sft import ROOT, read_json, sha256, atomic_json, encode
from echolens.candidate import CONFIG, validate_assets, convert, digest, cached_result, commit_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--limit',type=int,help='Engineering probe only; preserves completed records')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.dataset):
        raise ValueError('Invalid dataset ID')
    folder = ROOT/'data/workbench'/args.dataset
    config = validate_assets()
    metadata = read_json(folder/'dataset.json')
    assert metadata['analysis_config']==CONFIG
    assert metadata['snapshot_sha256']==sha256(folder/'records.jsonl'), 'Dataset changed'
    rows = [json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    identity = {'config':config,'snapshot_sha256':metadata['snapshot_sha256'],
        'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','peft','bitsandbytes','accelerate']},
        'source_files':{p:sha256(ROOT/p) for p in ['scripts/analyze_stage3.py','scripts/probe_model.py','src/echolens/candidate.py','src/echolens/stage3_sft.py']}}
    old = read_json(folder/'candidate-run.json')
    if old is not None and old!=identity:
        raise ValueError('候选运行代码、环境或配置改变，请创建新运行版本')
    atomic_json(folder/'candidate-run.json',identity)
    run_hash = digest(identity)
    done = {row['record_id'] for row in rows if cached_result(folder/'extractions'/f"{row['record_id']}.json",run_hash,row['text_raw']) is not None}
    todo = [row for row in rows if row['record_id'] not in done]
    if args.limit is not None:
        assert args.limit > 0
        todo = todo[:args.limit]
    def progress(status, **extra):
        atomic_json(folder/'progress.json',{'status':status,'phase':'candidate_extraction','pid':os.getpid(),
            'completed':len(done),'total':len(rows),**extra})
    progress('running')
    try:
        if todo:
            set_seed(config['seed'])
            tokenizer = AutoTokenizer.from_pretrained(ROOT/config['model']['local_dir'],local_files_only=True)
            prompt = (ROOT/config['prompt_path']).read_text(encoding='utf-8')
            aspects = read_json(ROOT/config['aspects_path'])
            model = None
            for row in todo:
                ids = encode(tokenizer,prompt,row['text_raw'])
                if len(ids)>config['max_input_tokens']:
                    result = convert(row,'',aspects,error='输入超过4096 tokens，未静默截断')
                else:
                    if model is None:
                        model = PeftModel.from_pretrained(load_model(ROOT/config['model']['local_dir']),ROOT/config['adapter']).eval()
                    inputs = torch.tensor([ids],device='cuda')
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                            max_new_tokens=config['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    tokens = generated[0,len(ids):].tolist()
                    result = convert(row,tokenizer.decode(tokens,skip_special_tokens=True),aspects,
                        hit_limit=len(tokens)>=config['max_new_tokens'],tokens=len(tokens),seconds=time.perf_counter()-started)
                    del inputs,generated
                result['input_tokens'] = len(ids)
                commit_result(folder/'extractions'/f"{row['record_id']}.json",result,run_hash,row['text_raw'])
                done.add(row['record_id'])
                progress('running')
                print(f"Candidate {len(done)}/{len(rows)} valid={result['valid']} seconds={result['seconds']:.1f}",flush=True)
        progress('complete' if len(done)==len(rows) else 'pilot_complete')
    except Exception as exc:
        progress('failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
