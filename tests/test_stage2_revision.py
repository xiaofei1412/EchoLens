import json
import numpy as np
import pytest
from echolens.opinions_v2 import merge_adjacent
from echolens.context_worker import windows, decode_window
from echolens.topic_groups import group_opinions, representative_units
from echolens.workbench import highlighted, review_reasons
from echolens.quote_extraction import locate_quote, decode_item


def item(**kwargs):
    return {"a":"口味", "s":1, "i":"praise", "t":"菌菇汤", "e":"菌菇汤不错！", **kwargs}


def test_context_quote_retains_dish_and_merges_repetition():
    text = "菌菇汤不错！真好喝！但烤南瓜里的薄荷很奇怪。"
    output = {"opinions":[item(), item(e="菌菇汤不错！真好喝！"), item(a="食材选择与搭配",s=-1,i="complaint",t="烤南瓜",e="但烤南瓜里的薄荷很奇怪。")]}
    opinions, rejected = decode_window(json.dumps(output,ensure_ascii=False), text, 6)
    assert not rejected
    opinions = merge_adjacent(opinions,text)
    assert len(opinions) == 2 and opinions[0]["raw_indices"] == [0,1]
    assert opinions[0]["evidence"] == "菌菇汤不错！真好喝！"
    assert opinions[1]["aspect"] == "Food#Ingredients"
    highlighted(text, opinions)


def test_distinct_dishes_not_merged():
    text = "菌菇汤不错！海带汤不错。"
    opinions, rejected = decode_window(json.dumps({"opinions":[item(),item(t="海带汤",e="海带汤不错。")]}),text,6)
    assert not rejected and len(merge_adjacent(opinions,text)) == 2


def test_merge_never_drops_earlier_context_in_the_same_clause():
    text = "不是菌菇汤好喝。"
    opinions, _ = decode_window(json.dumps({"opinions":[item(e="菌菇汤好喝。"),item(e=text)]}),text,6)
    assert merge_adjacent(opinions,text)[0]["evidence"] == text


@pytest.mark.parametrize("change", [{"t":123},{"e":"并不存在的原文"},{"a":"任意未知标签"},{"e":99},{"s":True}])
def test_invalid_fields_or_ungrounded_quote_rejected(change):
    with pytest.raises(ValueError):
        decode_item(item(**change),"菌菇汤不错！")


def test_unknown_stays_reviewable_not_classifier_index():
    op = decode_item(item(a="未覆盖 / 待定",t=""),"菌菇汤不错！")
    reasons = review_reasons("12345678", {"12345678":{"labels":[1]*18,"confidence":[1.0]*18}}, {"12345678":{"valid":True,"opinions":[op]}})
    assert "方面或对象待定" in reasons


def test_topics_cannot_mix_aspects_or_count_duplicates_as_support():
    units = [{"record_id":str(i),"aspect":aspect} for aspect in ("Food#Taste","Ambience#Decoration") for i in range(3)]
    units += [{"record_id":"single","aspect":"Service#Parking"} for _ in range(5)]
    vectors = np.tile([1.0,0.0],(len(units),1))
    labels = group_opinions(units,vectors,min_feedbacks=3)
    assert len(set(labels[:6])) == 2 and set(labels[6:]) == {-1}
    for label in set(labels)-{-1}:
        assert len({u["aspect"] for u,l in zip(units,labels) if l == label}) == 1


def test_representatives_use_distinct_records():
    units = [{"record_id":rid,"centrality":score} for rid,score in [("a",1), ("a",.99),("b",.9),("c",.8)]]
    assert [u["record_id"] for u in representative_units(units)] == ["a","b","c"]


def test_quote_alignment_restores_only_unique_source_spans():
    text = "推荐菌菇汤！每人喝了2碗。希望下次还有。"
    start,end,method = locate_quote("推荐菌菇汤！每人喝了 2 碗。",text)
    assert text[start:end] == "推荐菌菇汤！每人喝了2碗。" and method == "whitespace_restored"
    start,end,method = locate_quote("推荐菌菇汤...希望下次还有。",text)
    assert text[start:end] == text and method == "ellipsis_restored"
    with pytest.raises(ValueError):
        locate_quote("菌菇汤难喝",text)
    with pytest.raises(ValueError):
        locate_quote("很好吃","很好吃！很好吃！")


def test_quote_can_preserve_leading_and_trailing_source_whitespace():
    op = decode_item(item(t="汤",e=" 汤好喝。 "),"菜来了。 汤好喝。 ")
    assert op["evidence"] == " 汤好喝。 "
    assert (op["start"],op["end"],op["sentence_start"],op["sentence_end"]) == (4,10,1,1)


def test_consumption_does_not_become_portion_claim():
    raw = {"a":"分量","s":1,"i":"praise","t":"菌菇汤","e":"菌菇汤每人喝了2碗。"}
    with pytest.raises(ValueError,match="进食数量"):
        decode_item(raw,raw["e"])


def test_comparison_scope_is_not_clustered_as_current_shop():
    raw = {"a":"容易寻找","s":-1,"i":"complaint","t":"另一家店","e":"像另一家店真不好找"}
    op = decode_item(raw,raw["e"])
    assert op["scope"] == "uncertain" and op["review_flags"]
    units = [{"record_id":str(i),**op} for i in range(3)]
    assert set(group_opinions(units,np.tile([1.,0.],(3,1)))) == {-1}


def test_over_limit_retains_valid_items_and_exposes_failures():
    accepted, rejected = decode_window(json.dumps({"opinions":[item(),item(a="无效方面")]}),"菌菇汤不错！",1)
    assert len(accepted) == 1
    assert len(rejected) == 2 and "提示上限" in rejected[0]["error"]


def test_windows_preserve_offsets_and_context_on_both_sides():
    text = "汤很好喝。尤其是菌菇汤！但薄荷太多。希望改进。停车方便。"
    chunks = windows(text,chars=12,context=1)
    assert chunks[1]["offset"] < chunks[1]["core_start"]
    assert chunks[1]["offset"]+len(chunks[1]["text"]) > chunks[1]["core_end"]
    assert chunks[0]["core_start"] == 0 and chunks[-1]["core_end"] == len(text)
    assert all(a["core_end"] == b["core_start"] for a,b in zip(chunks,chunks[1:]))
    assert all(text[w["offset"]:w["offset"]+len(w["text"])] == w["text"] for w in chunks)
