"""Stage3 candidate contract: fail closed, preserve raw output, resume by identity."""
import hashlib
import json

from echolens.stage3_sft import ROOT, read_json, sha256, atomic_json, parse_output

CONFIG = 'configs/stage3-candidate.json'
SENTIMENT_CODES = {'negative':-1, 'neutral':0, 'positive':1, 'mixed':2, 'unknown':3}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def validate_assets():
    config = read_json(ROOT/CONFIG)
    for name in ['prompt','aspects']:
        if sha256(ROOT/config[name+'_path']) != config[name+'_sha256']:
            raise ValueError('候选模型提示或标签文件发生变化')
    for name,expected in config['adapter_files'].items():
        if sha256(ROOT/config['adapter']/name) != expected:
            raise ValueError('候选适配器文件发生变化')
    return config


def convert(row, output, aspects, *, hit_limit=False, seconds=0, tokens=0, error=None):
    checked = parse_output(output,row['text_raw'],aspects)
    reasons = [error] if error else []
    if hit_limit:
        reasons.append('达到输出长度上限，整条待复核')
    reasons.extend(checked['errors'])
    opinions = checked['valid']
    if any(o['aspect']=='unknown' or o['sentiment']=='unknown' or o['scope']=='uncertain' for o in opinions):
        reasons.append('方面、情感或归属不确定，整条待复核')
    keys = [(o['target'],o['aspect'],o['sentiment'],o['intent'],o['scope'],o['start'],o['end']) for o in opinions]
    if len(set(keys)) != len(keys):
        reasons.append('完全重复的意见，整条待复核')
    valid = not reasons and checked['json_valid'] and checked['schema_valid']
    accepted = [{**o,'sentiment':SENTIMENT_CODES[o['sentiment']],'issue':o['evidence']} for o in opinions] if valid else []
    return {'record_id':row['record_id'],'output':output,'seconds':seconds,'tokens':tokens,
        'hit_output_limit':hit_limit,'valid':valid,'fully_valid':valid,'partial':False,
        'opinions':accepted,'error':'; '.join(dict.fromkeys(reasons)) or None,
        'needs_review':not valid or not accepted,'empty_result':valid and not accepted,
        'rejected':reasons,'review_flags':[],'model_variant':'stage3-candidate-v1'}


def cached_result(path, run_hash, text):
    result = read_json(path)
    if result is None:
        return None
    content_hash = result.pop('content_sha256',None)
    if content_hash != digest(result) or result.get('run_sha256') != run_hash or result.get('input_sha256') != digest(text):
        raise ValueError('已有候选结果被修改或版本不一致，拒绝混用')
    result['content_sha256'] = content_hash
    return result


def commit_result(path, result, run_hash, text):
    payload = {**result,'run_sha256':run_hash,'input_sha256':digest(text)}
    atomic_json(path,{**payload,'content_sha256':digest(payload)})
