"""Constraint checks and transparent v2/v3 development comparison, never gold accuracy."""
import argparse
import html
import json
import hashlib
import statistics
from collections import Counter
from echolens.experiments import ROOT,sha256
from echolens.workbench import (dataset_path,records_at,results_at,read_json,atomic_json,
                               highlighted,topic_state,report_markdown,OPINION_LABELS,SENTIMENTS)
from echolens.topic_groups import representative_units
from verify_stage2_revision import summarize


def case_checks(rows, results):
    ops={r['source_record_id']:results.get(r['record_id'],{}).get('opinions',[]) for r in rows}
    def has(sid,aspect,sentiment,word):
        return any(o['aspect']==aspect and o['sentiment']==sentiment and word in o['evidence'] for o in ops[sid])
    return {
        '38689_decoration':has('38689','Ambience#Decoration',-1,'装修'),
        '6479_space':has('6479','Ambience#Space',-1,'桌子'),
        '44572_parking':has('44572','Service#Parking',-1,'停车'),
        '32524_parking':has('32524','Service#Parking',-1,'停车'),
        '5365_dish_context':sum(o['target']=='石锅海鲜饭' and o['sentiment']==1 and '重点推荐' in o['evidence'] and '果然没错' in o['evidence'] for o in ops['5365'])==1,
        '5365_ingredients':has('5365','Food#Ingredients',-1,'卷心菜') and not has('5365','Food#Portion',-1,'卷心菜'),
        '22304_no_unsupported_praise':bool(ops['22304']) and not any(o['sentiment']==1 and o['target'] in ('鸭子汤','口条') for o in ops['22304']),
        '26480_indirect_portion_complaint':has('26480','Food#Portion',-1,'五折'),
        '44206_transport_not_parking_queue':has('44206','Location#Transportation',1,'方便') and not any(o['aspect'] in ('Service#Parking','Service#Queue') for o in ops['44206']),
        '33803_temperature_not_wait':has('33803','Food#Taste',-1,'温') and not any(o['aspect']=='Service#Timely' and '温' in o['evidence'] for o in ops['33803']),
        '15748_preserve_negative_turn':has('15748','Food#Taste',-1,'腻'),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--partial',action='store_true'); args=ap.parse_args()
    old,new=dataset_path('asap-demo-1000-v2'),dataset_path('asap-demo-1000-v3')
    rows=records_at(new); by_id={r['record_id']:r for r in rows}; results=results_at(new); baseline=results_at(old)
    for ds,files in read_json(new/'baseline-preservation.json').items():
        for file,digest in files.items(): assert sha256(dataset_path(ds)/file)==digest
    assert sha256(old/'records.jsonl')==sha256(new/'records.jsonl')
    assert sha256(old/'classifications.json')==sha256(new/'classifications.json')
    source_digest=hashlib.sha256(''.join(sha256(p) for p in sorted((old/'extractions').glob('*.json'))).encode()).hexdigest()
    assert source_digest==read_json(new/'analysis-run.json')['source_extractions_sha256']
    reuse=read_json(new/'reused-reviews.json',{})
    if reuse:
        current=read_json(new/'analysis-run.json')
        for key in ('snapshot_sha256','models','classifier_sha256','prompt_sha256','source_extractions_sha256'):
            assert current[key]==reuse['source_run'][key]
        for key in ('quantized','enable_thinking','generation_batch_size','max_new_tokens','max_input_tokens'):
            assert current['config'][key]==reuse['source_run']['config'][key]
        for name,digest in reuse['files'].items():assert sha256(new/'semantic_reviews'/name)==digest
    if not args.partial: assert len(results)==len(rows)==1000
    for rid,r in results.items():
        highlighted(by_id[rid]['text_raw'],r['opinions'])
        for op in r['opinions']: assert op['target'] in op['evidence']
    selection=read_json(new/'audit-selection.json'); state=topic_state(new)
    if not args.partial:
        expected={f'{rid}:{i}' for rid,r in results.items() if r['valid'] for i in range(len(r['opinions']))}
        assert expected==set(state['assignments'])=={u['unit_id'] for u in state['units']}
        assert read_json(new/'job.json')['status']=='complete'
    support=[]
    for t in state['topics']:
        members=[u for u in state['units'] if state['assignments'][u['unit_id']]==t['id']]
        n=len({u['record_id'] for u in members}); support.append(n)
        assert n>=10 and len({u['aspect'] for u in members})==1
        assert all(u['scope']=='current' and u['aspect']!='unknown' for u in members)
        reps=representative_units(members); assert len({u['record_id'] for u in reps})==len(reps)
    checks=case_checks(rows,results)
    counts=Counter(state['assignments'].values()); names=Counter(t['name'] for t in state['topics'])
    old_topics=read_json(old/'topics.json')
    old_support=[len({u['record_id'] for u in old_topics['units'] if old_topics['assignments'][u['unit_id']]==t['id']}) for t in old_topics['topics']]
    report={'scope':'clean-train development only, no human-gold accuracy claimed','status':'partial' if args.partial else 'complete',
            'baseline_unchanged':True,'same_input_and_classifier':True,'case_checks':checks,
            'v2_same_processed_records':summarize(baseline,results),'v3':summarize(results,results),
            'candidate_count':sum(r.get('source_candidate_count',0) for r in results.values()),
            'discarded_candidates':sum(len(r.get('discarded',[])) for r in results.values()),
            'rule_adjusted_opinions':sum(bool(o.get('semantic_adjustments')) for r in results.values() for o in r['opinions']),
            'aspect_opinion_counts':{label:dict(Counter(o['aspect'] for rid in results for o in data.get(rid,{}).get('opinions',[]))) for label,data in [('v2',baseline),('v3',results)]},
            'additional_audit':summarize(results,selection['additional_audit']),
            'topics':len(state['topics']),'topic_sizes':dict(counts),'topic_feedback_counts':support,
            'topic_support_comparison':{label:{'topics':len(values),'under_10_feedbacks':sum(n<10 for n in values),'median_feedbacks':statistics.median(values) if values else None,'max_feedbacks':max(values,default=0)} for label,values in [('v2',old_support),('v3',support)]},
            'outlier_opinions':counts.get('-1',0),'duplicate_name_groups':sum(n>1 for n in names.values()),
            'topic_seconds':state.get('seconds'),'generation_batch_seconds':sum(r['seconds'] for r in results.values()),
            'limitations':['Case checks are development regressions after feedback, not independent evaluation',
                           'Schema and source spans do not prove semantic correctness','Average linkage constrains merge averages, not every pair of members',
                           'Sparse opinions remain ungrouped; low prevalence is not low importance',
                           'V3 reviews v2 candidates plus uncovered contrast clauses, not exhaustive fresh extraction',
                           'Source spans are copied by code; all-exact matching is not a model quoting accuracy score']}
    suffix='pilot' if args.partial else 'workflow'
    atomic_json(ROOT/f'docs/reports/stage2-v3-{suffix}.json',report)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>第三轮修订对照</title><style>body{font:16px/1.8 system-ui;max-width:1200px;margin:32px auto;padding:20px;color:#25363f;background:#f7f8fa}article{background:white;padding:20px;margin:16px 0}section{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre,blockquote{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{background:#edf3f5;padding:12px;margin:8px 0}nav a{margin-right:12px}@media(max-width:700px){section{grid-template-columns:1fr}}</style><h1>第三轮 v2 / v3 对照 · 待人工复核</h1><p>64 条固定训练样本，包括开发案例和另外固定样本。不是独立测试，不是人工金标准。</p>']
    ids=selection['pilot']+selection['additional_audit']
    parts.append('<nav>'+''.join(f'<a href="#r-{by_id[rid]["source_record_id"]}">{by_id[rid]["source_record_id"]}</a>' for rid in ids)+'</nav>')
    for rid in ids:
        row=by_id[rid]; parts.append(f'<article id="r-{row["source_record_id"]}"><h2>记录 {row["source_record_id"]}</h2><details><summary>完整原文</summary><pre>{html.escape(row["text_raw"])}</pre></details><section>')
        for label,data in [('v2',baseline.get(rid)),('v3',results.get(rid))]:
            parts.append(f'<div><h3>{label}</h3>')
            if data:
                parts.append('<p>'+('字段与定位通过，语义待审' if data.get('fully_valid') else '部分待复核或失败')+'</p>')
                for op in data['opinions']:
                    parts.append(f'<b>{OPINION_LABELS[op["aspect"]]} · {SENTIMENTS[op["sentiment"]]}</b><p>对象：{html.escape(op.get("target") or "待定")}</p><blockquote>{html.escape(op["evidence"])}</blockquote>')
                parts.append(f'<details><summary>完整结果与校验原因</summary><pre>{html.escape(json.dumps(data,ensure_ascii=False,indent=2))}</pre></details>')
            else: parts.append('<p>未运行</p>')
            parts.append('</div>')
        parts.append('</section></article>')
    parts.append('</html>')
    (ROOT/'artifacts/stage-02/revision-v3-comparison.html').write_text(''.join(parts),encoding='utf-8')
    if not args.partial:
        (ROOT/'artifacts/stage-02/feedback-report-v3.md').write_text(report_markdown(new,rows,read_json(new/'classifications.json'),results),encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('topic_sizes','topic_feedback_counts','v2_same_processed_records','additional_audit','aspect_opinion_counts')},ensure_ascii=False,indent=2))
    assert all(checks.values()),f'Development case checks failed: {[name for name,passed in checks.items() if not passed]}'


if __name__=='__main__': main()
