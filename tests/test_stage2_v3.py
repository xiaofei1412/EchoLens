import numpy as np
import json
import pytest
from echolens.quote_extraction import decode_item
from echolens.quote_extraction_v3 import deduplicate
from echolens.topic_groups import group_opinions
from echolens.semantic_review import apply_decision, candidates, evidence_constraints


def opinion(text, quote, target='', sentiment=1):
    return decode_item({'a':'口味','s':sentiment,'i':'praise' if sentiment==1 else 'complaint',
                        't':target,'e':quote},text)


def test_overlapping_unresolved_targets_count_once_but_polarity_is_preserved():
    text = '汤香浓好喝，不过后来觉得太咸。'
    ops = [opinion(text,'汤香浓好喝'),opinion(text,'香浓好喝'),
           opinion(text,text,sentiment=-1)]
    merged = deduplicate(ops,text)
    assert len(merged)==2
    assert merged[0]['sentiment']==-1 and merged[1]['sentiment']==1
    assert merged[1]['evidence']=='汤香浓好喝'


def test_different_explicit_objects_never_merge_even_with_shared_quote():
    text='鱼和虾都不错。'
    assert len(deduplicate([opinion(text,text,'鱼'),opinion(text,text,'虾')],text))==2


def test_dedup_restores_known_target_and_provenance():
    text='汤香浓好喝。'
    ops=[opinion(text,text),opinion(text,'汤香浓好喝','汤')]
    ops[1]['raw_indices']=[1]
    result=deduplicate(ops,text)
    assert len(result)==1 and result[0]['target']=='汤'
    assert result[0]['raw_indices']==[0,1] and result[0]['evidence']==text


def test_v3_main_topics_require_ten_distinct_feedbacks():
    units=[{'record_id':str(i),'aspect':'Food#Taste'} for i in range(10)]
    units += [{'record_id':str(i),'aspect':'Service#Parking'} for i in range(9)]
    vectors=np.tile([1.,0.],(len(units),1))
    labels=group_opinions(units,vectors,.42,10,'average')
    assert set(labels[:10])=={0} and set(labels[10:])=={-1}


def test_semantic_review_can_drop_factual_mention_without_faking_a_negative():
    text='点了鱼，还没吃。'
    draft=opinion(text,text,'鱼')
    op,decision=apply_decision(json.dumps({'reason':'仅提及点单','a':'菜品推荐','s':0,'discard':True,'t':'鱼'}),draft,text)
    assert op is None and decision['discard'] is True


def test_review_object_can_restore_nearby_context_and_never_invent_evidence():
    text='菌菇汤端来了。真好喝。'
    draft=opinion(text,'真好喝。')
    result,_=apply_decision(json.dumps({'reason':'评价前文汤','a':'口味','s':1,'discard':False,'t':'菌菇汤'}),draft,text)
    assert result['evidence']==text and result['target']=='菌菇汤'
    result,_=apply_decision(json.dumps({'reason':'错误对象','a':'口味','s':1,'discard':False,'t':'不存在的菜'}),draft,text)
    assert result['evidence']=='真好喝。' and result['target']=='' and result['review_flags']


@pytest.mark.parametrize('change',[{'discard':'false'},{'s':True},{'a':'不在标签表'},{'t':123}])
def test_invalid_semantic_review_is_not_accepted(change):
    text='汤好喝。';draft=opinion(text,text,'汤')
    payload={'reason':'有明确好评','a':'口味','s':1,'discard':False,'t':'汤',**change}
    with pytest.raises(ValueError):apply_decision(json.dumps(payload),draft,text)


def test_uncovered_contrast_is_only_a_candidate_not_an_assigned_negative():
    text='汤好喝。不过后来有点咸。'
    draft=opinion(text,'汤好喝。','汤')
    result=candidates(text,{'opinions':[draft]})
    assert len(result)==2 and result[1]['aspect']=='unknown' and result[1]['sentiment']==0


def test_aspect_evidence_constraints_do_not_infer_sentiment_or_use_record_ids():
    assert evidence_constraints('Service#Timely','汤是凉的，口感很差')=='Food#Taste'
    assert evidence_constraints('Service#Timely','等了半小时，汤都凉了')=='Service#Timely'
    assert evidence_constraints('Location#Transportation','停车场收费高')=='Service#Parking'
    assert evidence_constraints('Location#Transportation','公交直达')=='Location#Transportation'
    assert evidence_constraints('Location#Downtown','出门就是地铁，交通方便')=='Location#Transportation'
    assert evidence_constraints('Location#Downtown','商业街黄金地段')=='Location#Downtown'
    assert evidence_constraints('Food#Portion','推荐米线，吃了两碗')=='Food#Recommend'
    assert evidence_constraints('Food#Portion','这份米线量足，推荐')=='Food#Portion'


def test_contrast_gap_is_not_hidden_by_a_later_negative_in_same_sentence():
    text='不过有点腻，而且芝士风味淡。'
    draft=opinion(text,'而且芝士风味淡。','芝士',-1)
    result=candidates(text,{'opinions':[draft]})
    assert any(o['candidate_origin']=='uncovered_contrast' for o in result if 'candidate_origin' in o)


def test_known_candidate_offsets_disambiguate_repeated_contrast_text():
    text='不过太咸。不过太咸。'
    drafts=candidates(text,{'opinions':[]})
    assert len(drafts)==2 and drafts[1]['start']==5
    output=json.dumps({'reason':'抱怨咸','a':'口味','s':-1,'discard':False,'t':''})
    op,_=apply_decision(output,drafts[1],text)
    assert (op['start'],op['end'])==(5,10) and text[op['start']:op['end']]==op['evidence']
