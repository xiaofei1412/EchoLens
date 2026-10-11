import json
import pytest
from streamlit.testing.v1 import AppTest
from echolens import independent_review as review
from echolens.stage3_sft import atomic_json, sha256, ROOT


@pytest.fixture
def packet(tmp_path):
    folder = tmp_path/'packet'
    folder.mkdir()
    rows = [{'record_id':str(i),'source_record_id':str(i),'text':'奶油清爽，面包很软。','review_order':i+1} for i in range(2)]
    (folder/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    atomic_json(folder/'aspects.json',{'Food#Taste':'口味'})
    atomic_json(folder/'manifest.json',{'files':{n:sha256(folder/n) for n in ['records.jsonl','aspects.json']}})
    return folder


def payload():
    return {'actor':'synthetic fixture','status':'completed','choice':'after','coverage':True,
            'before_issues':['漏提意见'],'after_issues':[],'note':'合成测试'}


def test_review_requires_explicit_completion_and_preserves_history(packet):
    h = sha256(packet/'manifest.json')
    with pytest.raises(ValueError,match='读完整文'):
        review.save_review(packet,'0',h,'pair',{**payload(),'coverage':False})
    first = review.save_review(packet,'0',h,'pair',payload(),origin='automated_fixture')
    with pytest.raises(ValueError,match='另一窗口'):
        review.save_review(packet,'0',h,'pair',payload(),expected_revision=0)
    with pytest.raises(ValueError,match='结果已变化'):
        review.save_review(packet,'0',h,'different',payload(),expected_revision=first['revision'])
    second = review.save_review(packet,'0',h,'pair',{**payload(),'choice':'before'},expected_revision=first['revision'],origin='automated_fixture')
    assert len(review.history(packet))==2 and second['choice']=='before'
    assert not second['human_gold'] and not second['training_ready']


def test_invalid_prediction_stays_visible_and_partial_append_is_tolerated(tmp_path):
    result = {'record_id':'0','output':'{"opinions":[','hit_output_limit':True,'seconds':1}
    shown = review.view(result,'原文',{})
    assert shown['ready'] and not shown['checked']['json_valid'] and shown['raw']==result['output']
    (tmp_path/'predictions.jsonl').write_text(json.dumps(result)+'\n{"record',encoding='utf-8')
    assert list(review.predictions(tmp_path))==['0']


def test_pair_page_save_and_next_uses_only_synthetic_packet(packet,tmp_path,monkeypatch):
    run = tmp_path/'runs'
    op = {'target':'奶油','aspect':'Food#Taste','sentiment':'positive','intent':'praise','scope':'current','evidence':'奶油清爽','occurrence':1}
    for side in ['before','after']:
        (run/side).mkdir(parents=True)
        (run/side/'predictions.jsonl').write_text(''.join(json.dumps({'record_id':str(i),'output':json.dumps({'opinions':[op]},ensure_ascii=False),'hit_output_limit':False,'seconds':1})+'\n' for i in range(2)),encoding='utf-8')
    monkeypatch.setattr(review,'PACKET',packet)
    monkeypatch.setattr(review,'RUN',run)
    original = review.save_review
    monkeypatch.setattr(review,'save_review',lambda *args,**kwargs: original(*args,**kwargs,origin='automated_fixture'))
    app = AppTest.from_file(str(ROOT/'app/pages/4_独立样本核验.py')).run()
    assert not app.exception
    assert app.button[1].disabled  # Cannot complete while outputs are hidden.
    app.checkbox[0].check().run()
    assert not app.exception and not app.button[1].disabled
    app.selectbox[-1].set_value('after')
    app.checkbox[-1].check()
    app.button[1].click().run()
    assert not app.exception
    events = review.history(packet)
    assert len(events)==1 and events[0]['status']=='completed' and events[0]['origin']=='automated_fixture'
    assert app.selectbox[0].value==1
    # Malformed labels must stay visible instead of crashing the review page.
    broken = {**op,'aspect':['invalid'],'sentiment':{'bad':'value'}}
    for side in ['before','after']:
        (run/side/'predictions.jsonl').write_text(json.dumps({'record_id':'1','output':json.dumps({'opinions':[broken]}),
            'hit_output_limit':False,'seconds':1})+'\n',encoding='utf-8')
    app.checkbox[0].check().run()
    assert not app.exception
    assert any('无效字段' in warning.value for warning in app.warning)
