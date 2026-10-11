"""Frozen pairwise review packets and append-only human judgments, never training labels."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from echolens.stage3_sft import ROOT, read_json, sha256, parse_output, occurrences

PACKET = ROOT/'data/evaluation/stage3-independent50-v1'
RUN = ROOT/'artifacts/stage-03/independent50-v1'
ISSUES = ['漏提意见', '凭空新增/过度推断', '对象或范围错误', '方面错误', '情感/意图错误',
          '证据边界/定位错误', '重复或不当合并', '格式失败/输出截断']
CHOICES = {'unreviewed':'未选择', 'after':'微调后更好', 'before':'微调前更好',
           'tie_good':'两版接近且基本可用', 'tie_bad':'两版接近但都有明显问题', 'uncertain':'暂无法判断'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def load_packet(folder=PACKET):
    manifest = read_json(folder/'manifest.json')
    if not manifest:
        raise ValueError('独立样本包尚未准备')
    if not all(sha256(folder/p) == h for p,h in manifest['files'].items()):
        raise ValueError('冻结样本文件发生变化，请勿混用审核结果')
    rows = [json.loads(x) for x in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    return manifest, rows, read_json(folder/'aspects.json')


def predictions(folder):
    path = folder/'predictions.jsonl'
    if not path.exists():
        return {}
    lines = path.read_text(encoding='utf-8').splitlines()
    result = {}
    for i,line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1:  # An inference process may still be appending the last line.
                break
            raise
        if row['record_id'] in result:
            raise ValueError('预测记录重复，不能审核')
        result[row['record_id']] = row
    return result


def view(result, text, aspects):
    if result is None:
        return {'ready':False, 'ops':[], 'valid':[], 'error':'尚未生成'}
    checked = parse_output(result['output'],text,aspects)
    try:
        payload = json.loads(result['output'])
        ops = payload.get('opinions',[]) if isinstance(payload,dict) else []
        if not isinstance(ops,list):
            ops = []
    except (ValueError,TypeError):
        ops = []
    located = {}
    for number,op in enumerate(ops,1):
        if not isinstance(op,dict) or not isinstance(op.get('evidence'),str) or type(op.get('occurrence')) is not int:
            continue
        positions = occurrences(text,op['evidence'])
        if 1 <= op['occurrence'] <= len(positions):
            start = positions[op['occurrence']-1]
            located[number] = {'start':start,'end':start+len(op['evidence']),'evidence':op['evidence']}
    return {'ready':True, 'ops':ops, 'valid':checked['valid'], 'located':located, 'checked':checked,
            'raw':result['output'], 'hit_limit':result['hit_output_limit'], 'seconds':result['seconds']}


def history(folder=PACKET):
    path = folder/'reviews.sqlite3'
    if not path.exists():
        return []
    with sqlite3.connect(path) as db:
        return [{**json.loads(payload),'revision':revision} for revision,payload in db.execute('SELECT id,payload FROM events ORDER BY id')]


def save_review(folder, record_id, packet_hash, pair_hash, review, expected_revision=0, origin='human_ui'):
    manifest, rows, _ = load_packet(folder)
    if record_id not in {r['record_id'] for r in rows} or sha256(folder/'manifest.json') != packet_hash:
        raise ValueError('记录或冻结样本版本不一致')
    if review['choice'] not in CHOICES or review['status'] not in {'draft','completed'}:
        raise ValueError('审核状态无效')
    if any(x not in ISSUES for key in ['before_issues','after_issues'] for x in review[key]):
        raise ValueError('问题类型无效')
    if not review['actor'].strip():
        raise ValueError('请填写审核者')
    if review['status'] == 'completed':
        if not review['coverage'] or review['choice'] == 'unreviewed':
            raise ValueError('请读完整文、检查两版结果，并选择总体判断')
        if review['choice'] == 'uncertain' and not review['note'].strip():
            raise ValueError('无法判断时请简述疑问')
    payload = {**review, 'record_id':record_id,'packet_sha256':packet_hash,'pair_sha256':pair_hash,
        'origin':origin,'saved_at':datetime.now(timezone.utc).isoformat(),
        'usage':'independent_pairwise_review_only','training_ready':False,'human_gold':False}
    with sqlite3.connect(folder/'reviews.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY,record_id TEXT NOT NULL,payload TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        latest = db.execute('SELECT id,payload FROM events WHERE record_id=? ORDER BY id DESC LIMIT 1',(record_id,)).fetchone()
        if (latest[0] if latest else 0) != expected_revision:
            raise ValueError('另一窗口已保存此条，请重新载入后再修改')
        if latest:
            old = json.loads(latest[1])
            if old['packet_sha256'] != packet_hash or old['pair_sha256'] != pair_hash:
                raise ValueError('原文或模型结果已变化，不能混写审核历史')
        cursor = db.execute('INSERT INTO events(record_id,payload) VALUES (?,?)',(record_id,json.dumps(payload,ensure_ascii=False)))
        payload['revision'] = cursor.lastrowid
    return payload
