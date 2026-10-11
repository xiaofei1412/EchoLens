"""Small Streamlit pilot editor; never writes stage 2 predictions."""
import json
import uuid
from collections import Counter

import streamlit as st

from echolens.annotation import (PACKET, GUIDELINE, SENTIMENTS, INTENTS, SCOPES, STATUSES,
    digest, load_packet, occurrences, latest, history, save_annotation)
from echolens.experiments import sha256
from echolens.workbench import highlighted, sentences


def card(op=None):
    return {"aspect": "unknown", "sentiment": "unknown", "intent": "unknown", "target": "",
            "scope": "current", "evidence": "", "start": None, "end": None,
            **(op or {}), "_key": uuid.uuid4().hex}


def clean(ops):
    return [{k: v for k, v in op.items() if not k.startswith("_")} for op in ops]


def render_annotation_page(packet_path=PACKET):
    st.markdown("<style>.block-container {padding:2rem;max-width:1500px}.evidence-text {white-space:pre-wrap;line-height:1.9;padding:18px;background:white;border:1px solid #dfe5e8;border-radius:10px;overflow-wrap:anywhere} mark {background:#ffefb5;color:#25363f}</style>", unsafe_allow_html=True)
    st.title("阶段3 · 人工试标")
    st.caption("先试标20条，统一标准后再扩大到200条。本页不训练模型，也不把草稿当作人工金标准。")
    try:
        packet = load_packet(packet_path)
    except ValueError as exc:
        st.info(str(exc))
        return
    if sha256(GUIDELINE) != packet["guideline_sha256"]:
        st.error("指南与试标包版本不一致，请先建立新版本并确认重标范围。")
        return
    folder = packet_path.parent
    saved = latest(folder)
    counts = Counter(e["status"] for e in saved.values())
    st.progress(counts["completed"] / len(packet["records"]),
                text=f"已完整核验 {counts['completed']} / {len(packet['records'])} · 待讨论 {counts['needs_discussion']} · 草稿 {counts['draft']}")
    actor = st.text_input("标注者（昵称即可）", value="本地标注者", key="annotation_actor")
    with st.expander("先看操作与标注指南"):
        st.markdown(GUIDELINE.read_text(encoding="utf-8"))
    prefix = packet["packet_sha256"]
    index_key = f"annotation_index_{prefix}"
    if index_key not in st.session_state:
        st.session_state[index_key] = next((i for i, r in enumerate(packet["records"]) if saved.get(r["record_id"], {}).get("status") != "completed"), 0)
    index = st.session_state[index_key]
    row = packet["records"][index]
    rid = row["record_id"]
    draft_key = f"annotation_draft_{prefix}_{rid}"
    if draft_key not in st.session_state:
        previous = saved.get(rid, {})
        st.session_state[draft_key] = {"ops": [card(o) for o in previous.get("opinions", row["suggestions"])],
            "note": previous.get("note", ""), "revision": previous.get("revision", 0),
            "coverage": previous.get("coverage_confirmed", False), "empty": previous.get("no_opinions_confirmed", False)}
    draft = st.session_state[draft_key]
    st.subheader(f"第 {index + 1} / {len(packet['records'])} 条 · 来源记录 {row['source_record_id']}")
    st.caption(f"已保存状态：{STATUSES.get(saved.get(rid, {}).get('status'), '未标注')} · 训练池试标 · 指南 {packet['guideline_version']}")
    left, right = st.columns([1, 1.2])
    text = row["text_raw"]
    with left:
        st.markdown("**完整原文**")
        valid_spans = [o for o in draft["ops"] if type(o.get("start")) is int and type(o.get("end")) is int
                       and 0 <= o["start"] < o["end"] <= len(text) and text[o["start"]:o["end"]] == o["evidence"]]
        st.markdown(highlighted(text, valid_spans), unsafe_allow_html=True)
        st.caption("高亮仅表示位置匹配，不表示语义已通过。请从完整原文检查遗漏。")
        with st.expander("选取句段，新增遗漏意见"):
            spans = sentences(text)
            first = st.selectbox("起始句段", range(len(spans)), format_func=lambda i: f"{i+1}. {spans[i]['text']}", key=f"first_{draft_key}")
            last = st.selectbox("结束句段", range(first, len(spans)), format_func=lambda i: f"{i+1}. {spans[i]['text']}", key=f"last_{draft_key}_{first}")
            if st.button("从选中句段新增意见", key=f"span_{draft_key}"):
                start, end = spans[first]["start"], spans[last]["end"]
                draft["ops"].append(card({"evidence": text[start:end], "start": start, "end": end}))
                st.rerun()
        st.info("草稿可能漏标或判断错误。一个方面一条意见；同一证据可支撑不同方面。无关背景不需要标成中性意见。")
    with right:
        st.markdown("**意见卡片 · 请核验模型草稿**")
        for n, op in enumerate(draft["ops"], 1):
            key = op["_key"]
            with st.expander(f"意见 {n} · {packet['aspects'].get(op['aspect'], '待定')} · {SENTIMENTS.get(op['sentiment'], '待定')}", expanded=n <= 2):
                op["evidence"] = st.text_area("原文证据（可粘贴原文，不能改写）", value=op["evidence"], key=f"e_{key}")
                matches = occurrences(text, op["evidence"])
                if len(matches) == 1:
                    op["start"] = matches[0]
                elif len(matches) > 1:
                    choices = [None] + matches
                    chosen = op["start"] if op["start"] in matches else None
                    op["start"] = st.selectbox("这段证据在原文中重复，请选出现位置", choices,
                        index=choices.index(chosen), format_func=lambda i, size=len(op["evidence"]): "请选择" if i is None else f"位置 {i}：…{text[max(0,i-12):i+size+12]}…",
                        key=f"occ_{key}_{digest(op['evidence'])}")
                else:
                    op["start"] = None
                    st.warning("当前证据未匹配原文；可以存草稿，完整核验前需要修正。")
                op["end"] = op["start"] + len(op["evidence"]) if op["start"] is not None else None
                if op["start"] is not None:
                    st.caption(f"已定位原文 [{op['start']}, {op['end']})")
                op["target"] = st.text_input("评价对象", value=op["target"], placeholder="如：菠萝海鲜饭；无法确定可明确填写", key=f"target_{key}")
                for field, label, options in (("aspect", "评价方面", packet["aspects"]), ("sentiment", "情感", SENTIMENTS), ("intent", "表达意图", INTENTS), ("scope", "评价归属", SCOPES)):
                    values = list(options)
                    op[field] = st.selectbox(label, values, index=values.index(op[field]), format_func=lambda x, opts=options: opts[x], key=f"{field}_{key}")
                a, b = st.columns(2)
                if a.button("复制后拆分", key=f"copy_{key}"):
                    draft["ops"].insert(n, card(clean([op])[0]))
                    st.rerun()
                if b.button("删除此意见", key=f"delete_{key}"):
                    draft["ops"].remove(op)
                    st.rerun()
        if st.button("新增空白意见", key=f"add_{draft_key}"):
            draft["ops"].append(card())
            st.rerun()
        with st.expander("合并同一对象、同一方面的重复意见"):
            chosen = st.multiselect("选择意见编号", range(len(draft["ops"])), format_func=lambda i: f"意见 {i+1}", key=f"merge_select_{draft_key}_{len(draft['ops'])}")
            if st.button("合并选中意见", key=f"merge_{draft_key}"):
                ops = [draft["ops"][i] for i in chosen]
                fields = ("aspect", "sentiment", "intent", "target", "scope")
                if len(ops) < 2 or any(o["start"] is None for o in ops) or len({tuple(o[f] for f in fields) for o in ops}) != 1:
                    st.error("至少选择两条已定位、对象及标签一致的意见；不同方面应分别保留。")
                else:
                    start, end = min(o["start"] for o in ops), max(o["end"] for o in ops)
                    merged = card({**clean([ops[0]])[0], "start": start, "end": end, "evidence": text[start:end]})
                    draft["ops"] = [o for i, o in enumerate(draft["ops"]) if i not in chosen] + [merged]
                    st.rerun()
            st.caption("合并保留两段之间的完整原文，请再次检查是否引入多余上下文。")
    draft["note"] = st.text_area("备注 / 待讨论的问题", value=draft["note"], key=f"note_{draft_key}")
    draft["coverage"] = st.checkbox("我已读完整条原文，检查了遗漏、重复、评价对象及标签", value=draft["coverage"], key=f"coverage_{draft_key}")
    draft["empty"] = st.checkbox("本条原文没有适用意见（仅在删除全部意见后使用）", value=draft["empty"], key=f"empty_{draft_key}")
    st.caption("编辑后请保存再关闭浏览器；完整核验表示单人审核完成，仍是试标数据，未进入训练。")

    def persist(status):
        try:
            event = save_annotation(folder, packet, rid, clean(draft["ops"]), status, actor, draft["note"],
                draft["coverage"], draft["empty"], draft["revision"],
                origin="automated_fixture" if packet.get("synthetic_fixture") else "human_ui")
            draft["revision"] = event["revision"]
            return True
        except ValueError as exc:
            st.error(str(exc))
            return False

    a, b, c = st.columns(3)
    if a.button("保存草稿", width="stretch") and persist("draft"):
        st.rerun()
    if b.button("暂无法判断并下一条", width="stretch") and persist("needs_discussion"):
        st.session_state[index_key] = min(index + 1, len(packet["records"]) - 1)
        st.rerun()
    if c.button("保存完整核验并下一条", type="primary", width="stretch") and persist("completed"):
        st.session_state[index_key] = min(index + 1, len(packet["records"]) - 1)
        st.rerun()
    a, b = st.columns(2)
    for col, label, target in ((a, "上一条", index - 1), (b, "下一条", index + 1)):
        if col.button(label + "（未保存编辑存为草稿）", disabled=not 0 <= target < len(packet["records"]), width="stretch"):
            previous = saved.get(rid, {})
            unchanged = (clean(draft["ops"]) == previous.get("opinions") and draft["note"] == previous.get("note")
                         and draft["coverage"] == previous.get("coverage_confirmed") and draft["empty"] == previous.get("no_opinions_confirmed"))
            if unchanged or persist("draft"):
                st.session_state[index_key] = target
                st.rerun()
    with st.expander("保存历史与备份"):
        entries = history(folder, rid)
        st.write([{k: e[k] for k in ("revision", "status", "actor", "saved_at")} for e in entries])
        st.download_button("下载全部标注历史（含原文，本地保管）", json.dumps({"packet": packet, "events": history(folder)}, ensure_ascii=False, indent=2),
                           file_name="echolens-stage3-pilot-backup.json", mime="application/json")
