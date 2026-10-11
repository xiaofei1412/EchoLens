import json
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from echolens import workbench as w
from echolens import candidate_workspace as cw
from echolens.candidate import CONFIG, convert, commit_result, cached_result
from echolens.topic_groups import group_opinions

LABELS = {'General#Facilities':'设施与器具适用性','Food#Taste':'口味','unknown':'未覆盖 / 待定'}
TEXT = '空调够凉。去年味道很好。今年味道有好有坏。'


def op(**changes):
    return {'target':'店内空调','aspect':'General#Facilities','sentiment':'positive','intent':'praise',
            'scope':'current','evidence':'空调够凉','occurrence':1,**changes}


def result(ops=None,**kwargs):
    return convert({'record_id':'a'*64,'text_raw':TEXT},json.dumps({'opinions':[op()] if ops is None else ops}),LABELS,**kwargs)


@pytest.mark.parametrize('change',[{'evidence':'编造的引用'},{'aspect':'bad'},{'sentiment':'unknown'},{'scope':'uncertain'},{'occurrence':2}])
def test_failures_never_enter_stats_or_topics(change):
    value = result([op(),op(**change)])
    assert not value['valid'] and not value['opinions'] and value['needs_review']
    assert not w.opinion_counts([{'record_id':'a'*64}],{'a'*64:value})
    assert '抽取失败' in w.review_reasons('a'*64,{}, {'a'*64:value})


def test_cap_invalid_json_empty_and_duplicate():
    assert not result(hit_limit=True)['valid']
    assert not result([op(),op()])['valid']
    assert not convert({'record_id':'a','text_raw':TEXT},'{',LABELS)['valid']
    empty = result([])
    assert empty['valid'] and empty['needs_review'] and empty['empty_result']
    assert '空结果需检查遗漏' in w.review_reasons('a'*64,{}, {'a'*64:empty})


def test_candidate_labels_do_not_change_legacy_prompts():
    from echolens import quote_extraction, semantic_review
    assert len(quote_extraction.OPINION_LABELS)==len(semantic_review.OPINION_LABELS)==21
    assert 'General#Facilities' not in quote_extraction.OPINION_LABELS
    assert '设施与器具适用性' not in semantic_review.PROMPT


def test_new_labels_mixed_and_past_survive_without_miscount():
    value = result([op(),op(aspect='Food#Taste',scope='past',evidence='去年味道很好'),
                    op(aspect='Food#Taste',sentiment='mixed',evidence='今年味道有好有坏')])
    assert value['valid'] and value['opinions'][2]['sentiment']==2
    counts = {c['aspect']:c for c in w.opinion_counts([{'record_id':'a'*64}],{'a'*64:value})}
    assert counts['Food#Taste']['混合']==1 and counts['Food#Taste']['正面']==0
    units = [{'record_id':str(i),**o} for i,o in enumerate(value['opinions'])]
    labels = group_opinions(units,np.eye(3),min_feedbacks=2)
    assert labels[1]==-1


def test_atomic_checkpoint_resume_and_tamper(tmp_path):
    path = tmp_path/'one.json'
    commit_result(path,result(),'run',TEXT)
    before = path.read_bytes()
    (tmp_path/'two.json.tmp').write_text('{',encoding='utf-8') # interrupted uncommitted write
    assert cached_result(tmp_path/'two.json','run',TEXT) is None
    assert cached_result(path,'run',TEXT)['valid'] and path.read_bytes()==before
    with pytest.raises(ValueError):
        cached_result(path,'different',TEXT)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['output']='changed'
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError):
        cached_result(path,'run',TEXT)


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'STORE',tmp_path)
    raw = tmp_path/'source.jsonl'
    raw.write_text(json.dumps({'id':'1','text':TEXT},ensure_ascii=False)+'\n',encoding='utf-8')
    source,_ = w.import_upload(raw,{'source_record_id':'id','text_raw':'text'},'test','v1','owned_or_synthetic')
    target = cw.create_candidate(source)
    return w.dataset_path(source),w.dataset_path(target)


def test_separate_versions_export_and_review_contract(tmp_path,monkeypatch):
    source,folder = fixture(tmp_path,monkeypatch)
    original = (source/'dataset.json').read_bytes()
    assert cw.create_candidate(source.name)==folder.name
    rows = w.records_at(folder)
    rid = rows[0]['record_id']
    value = result([op(aspect='Food#Taste',scope='past',sentiment='mixed',evidence='去年味道很好')])
    value['record_id']=rid
    assert '混合' in w.report_markdown(folder,rows,{}, {rid:value})
    w.save_review(folder,rid,value['opinions'],'approved','synthetic test','tester')
    assert not w.events(source) and (source/'dataset.json').read_bytes()==original
    failed = result(hit_limit=True)
    report = w.report_markdown(source,rows,{}, {rid:failed})
    assert '原文位置' not in report


def test_candidate_ui_pages_and_failure_queue(tmp_path,monkeypatch):
    _,folder = fixture(tmp_path,monkeypatch)
    row = w.records_at(folder)[0]
    value = result([op(aspect='Food#Taste',scope='past',sentiment='mixed',evidence='去年味道很好')])
    value['record_id']=row['record_id']
    (folder/'extractions').mkdir()
    w.atomic_json(folder/'extractions'/f"{row['record_id']}.json",value)
    app = AppTest.from_file(str(w.ROOT/'app/streamlit_app.py')).run()
    app.sidebar.selectbox[0].set_value(folder.name).run()
    next(b for b in app.button if b.label=='返回原分析版本').click().run()
    next(b for b in app.button if b.label=='创建 / 打开微调分析版本').click().run()
    assert app.sidebar.selectbox[0].value==folder.name and not app.exception
    assert not any('default value' in str(x.value) for x in app.warning)
    app.sidebar.selectbox[0].set_value(None).run()
    assert not app.exception and any(x.value=='请在侧栏选择数据集。' for x in app.info)
    app.sidebar.selectbox[0].set_value(folder.name).run()
    for page in ['分析概览','主题与证据','人工复核','报告导出']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    failed = result(hit_limit=True)
    failed['record_id']=row['record_id']
    w.atomic_json(folder/'extractions'/f"{row['record_id']}.json",failed)
    app.sidebar.radio[0].set_value('人工复核').run()
    assert not app.exception and any('抽取失败' in x.value for x in app.warning)
    assert not w.events(folder)
