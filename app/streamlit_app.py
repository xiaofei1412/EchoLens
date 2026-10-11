"""EchoLens local workbench. Model inference runs in a separate background worker."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from echolens.experiments import ROOT
from echolens.workbench import (STORE, ASPECTS, TAXONOMY, SENTIMENTS, OPINION_LABELS, datasets, dataset_path,
    read_json, records_at, results_at, highlighted, sentences, decode_opinions, aspect_counts,
    review_reasons, latest_reviews, save_review, events, topic_state, edit_topics,
    report_markdown, import_upload, job_running, opinion_counts, LEGACY_OPINION_LABELS)
from echolens.candidate_workspace import create_candidate
from echolens.runtime import missing_analysis_assets, start_analysis
from echolens.workbench_catalog import visible_datasets, dataset_label, preferred_dataset
from echolens.opinions_v2 import clauses
from echolens.topic_groups import representative_units

st.set_page_config(page_title="回声 EchoLens", page_icon="◉", layout="wide")
st.markdown("""<style>
.stApp {background:#f7f8fa} [data-testid="stSidebar"] {background:#edf1f3}
.block-container {padding-top:2rem;max-width:1500px} h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #dfe5e8;border-radius:12px;padding:14px}
[data-testid="stMetricValue"] {font-size:1.5rem}
.evidence-text {white-space:normal;line-height:1.9;background:white;padding:20px;border:1px solid #e1e7e9;border-radius:12px;overflow-wrap:anywhere}
mark {background:#ffefb5;color:#26333a;border-radius:3px;padding:1px 0}
</style>""", unsafe_allow_html=True)
st.sidebar.title("◉ 回声 EchoLens")
st.sidebar.caption("中文餐饮评价 · v0.1.0")
all_options = datasets()
include_history = st.sidebar.checkbox('显示历史与工程数据',value=False)
options = visible_datasets(all_options,include_history)
names = {m['dataset_id']:dataset_label(m) for m in options}
if st.session_state.get('open_analysis') in names:
    st.session_state['dataset'] = st.session_state.pop('open_analysis')
if st.session_state.get('dataset') is not None and st.session_state['dataset'] not in names:
    del st.session_state['dataset']
selected = None
if options:
    default_dataset = preferred_dataset(options)
    selected = st.sidebar.selectbox('当前数据集',list(names),index=None if 'dataset' in st.session_state else list(names).index(default_dataset),format_func=names.get,key='dataset')
workspaces = ['数据导入','分析概览','主题与证据','人工复核','报告导出']
if 'open_workspace' in st.session_state:
    st.session_state['workspace'] = st.session_state.pop('open_workspace')
if st.session_state.get('workspace') not in workspaces:
    st.session_state['workspace'] = '分析概览' if options else '数据导入'
page = st.sidebar.radio('工作区',workspaces,key='workspace')
with st.sidebar.expander('研究工具'):
    st.caption('用于历史实验和标注，不是日常分析的必经步骤。')
    st.page_link('pages/3_人工标注.py',label='人工标注工具')
    st.page_link('pages/4_独立样本核验.py',label='微调前后对照')
st.sidebar.divider()
st.sidebar.caption("本地处理 · 原文可追溯\n\n模型判断需核验，人工修改单独留痕。")
if 'workbench_notice' in st.session_state:
    st.success(st.session_state.pop('workbench_notice'))


def import_page():
    st.title("数据导入")
    st.caption("CSV / JSONL · 原文保留 · 相同来源版本重复导入不增加记录")
    uploaded = st.file_uploader("选择反馈文件", type=["csv", "jsonl"])
    if not uploaded:
        st.info("至少需要一列反馈原文和一列稳定的记录 ID；支持可选的 1–5 星评分。")
        return
    raw = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    incoming = ROOT / "data/imports/uploads" / (hashlib.sha256(raw).hexdigest() + suffix)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    incoming.write_bytes(raw)
    try:
        preview = pd.read_csv(incoming, dtype=str).fillna("") if suffix == ".csv" else pd.read_json(incoming, lines=True, dtype=False)
    except Exception as exc:
        st.error(f"文件无法读取：{exc}")
        return
    st.dataframe(preview.head(5), hide_index=True, width="stretch")
    columns = list(preview.columns)
    if not columns:
        st.error("文件没有字段")
        return
    with st.form("import"):
        source = st.text_input("来源名称", value="我的反馈")
        revision = st.text_input("数据版本", value="v1")
        text_field = st.selectbox("反馈原文字段", columns, index=columns.index("review") if "review" in columns else 0)
        id_field = st.selectbox("记录 ID 字段", columns, index=columns.index("id") if "id" in columns else 0)
        rating = st.selectbox("评分字段（可选）", ["不提供"] + columns)
        rights = st.selectbox("数据使用依据", ["用途尚未确认", "已获本地分析授权", "自有或合成数据"])
        analysis = st.selectbox('分析方式',['餐饮微调模型','原始模型基线'])
        st.caption('导入不会立即启动推理；在分析概览确认资源和数据后启动。')
        submitted = st.form_submit_button("导入并检查", type="primary")
    if submitted:
        if not source.strip() or not revision.strip() or text_field == id_field:
            st.error("请填写来源、版本，并选择不同的原文与 ID 字段。")
            return
        mapping = {"source_record_id": id_field, "text_raw": text_field}
        if rating != "不提供":
            mapping["rating"] = rating
        try:
            dataset_id, stats = import_upload(incoming, mapping, source.strip(), revision.strip(), {"用途尚未确认": "unknown", "已获本地分析授权": "authorized_analysis", "自有或合成数据": "owned_or_synthetic"}[rights])
            if analysis=='餐饮微调模型':
                dataset_id = create_candidate(dataset_id)
            st.session_state['open_analysis'] = dataset_id
            st.session_state['open_workspace'] = '分析概览'
            st.session_state['workbench_notice'] = f"导入完成：{stats['valid_unique_ids']}条有效记录，排除{len(stats['rejected'])}条，新增{stats.get('new_records',0)}条。尚未启动推理。"
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))


if page == "数据导入":
    import_page()
    st.stop()

if selected is None:
    st.info('请在侧栏选择数据集。' if options else '暂无可展示的数据，请先在“数据导入”中添加餐饮评价；历史与工程数据可在侧栏展开。')
    st.stop()

folder = dataset_path(selected)
metadata = read_json(folder / "dataset.json")
candidate = metadata.get('analysis_config')=='configs/stage3-candidate.json'
contextual = candidate or metadata.get("analysis_config") in ("configs/stage2-v2.json", "configs/stage2-v3.json")
st.sidebar.caption('当前模型：餐饮微调 · 第102步' if candidate else '当前模型：原始模型基线')
with st.sidebar.expander('选择分析版本'):
    if not candidate:
        st.caption('创建独立候选版本，保留当前结果与审核记录；创建后在概览启动分析。')
        if st.button('创建 / 打开微调分析版本'):
            try:
                st.session_state['open_analysis'] = create_candidate(selected)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    elif metadata.get('baseline_dataset'):
        if st.button('返回原分析版本'):
            st.session_state['open_analysis'] = metadata['baseline_dataset']
            st.rerun()
if metadata['kind']=='integration_probe':
    st.info('阶段3工程验证样例：来自开发数据，仅验证接入流程，不作为新的质量评测。')
if metadata["kind"] == "synthetic_fixture":
    st.warning("合成验收样例：预设分析仅用于测试页面，不是模型成绩或人工金标准。")
records = records_at(folder)
by_id = {r["record_id"]: r for r in records}
classifications = read_json(folder / "classifications.json", {})
extractions = results_at(folder)
reviews = latest_reviews(folder)
progress = read_json(folder / "progress.json", {"status": "not_started"})
state = topic_state(folder)
if page=='主题与证据':
    evidence_view = st.radio('查看内容',['主题浏览','原文检索'],horizontal=True,key='evidence_view')
    page = '主题候选' if evidence_view=='主题浏览' else '反馈与证据'


def show_record(row):
    rid = row["record_id"]
    result = extractions.get(rid)
    opinions = result["opinions"] if result and result["valid"] else []
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader(f"原文 · {row['source_record_id']}")
        st.markdown(highlighted(row["text_raw"], opinions), unsafe_allow_html=True)
        st.caption(f"来源：{row['source']} · 评分：{row['rating'] if row['rating'] is not None else '未提供'} · 日期：{row['created_at'] or '未提供'}")
        with st.expander("原文片段编号（用于审核）"):
            st.dataframe([{"编号": s["id"], "起始字符":s["start"], "结束字符":s["end"], "原文片段": s["text"]} for s in (clauses(row["text_raw"]) if contextual else sentences(row["text_raw"]))], hide_index=True, width="stretch")
    with right:
        st.subheader("机器分析")
        if not result:
            st.info("尚未完成意见抽取。")
        elif not result["valid"]:
            st.warning(f"抽取失败，需复核：{result['error']}")
            with st.expander("查看模型原始输出"):
                st.code(result.get("output", "未生成"), language="json")
        elif not opinions:
            st.info("模型未提取适用意见；不等于原文没有问题。")
        else:
            if result.get("partial"):
                st.warning(f"部分意见待复核：{len(result.get('rejected', []))} 项校验警示或失败，另有 {len(result.get('review_flags', []))} 个对象或归属提示。下方展示已定位的意见，不代表整条通过。")
                with st.expander("查看未通过的意见与原因"):
                    st.json({"未通过":result["rejected"],"对象提示":result.get("review_flags",[])})
            for index, op in enumerate(opinions, 1):
                st.markdown(f"**{index}. {OPINION_LABELS[op['aspect']]} · {SENTIMENTS[op['sentiment']]}**")
                if "target" in op:
                    st.text(f"评价对象：{op['target'] or '待定（不推定为整个餐厅）'}")
                    if op["issue"] != op["evidence"]:
                        st.text("具体意见：" + op["issue"])
                    if op.get("review_flags"):
                        st.caption("；".join(op["review_flags"]))
                    if op.get("grounding_method") in ("whitespace_restored","ellipsis_restored"):
                        st.caption("引用已按唯一匹配还原为原文；模型原始引文保留在运行记录中。")
                    if len(op.get("raw_indices", [])) > 1:
                        st.caption(f"已合并 {len(op['raw_indices'])} 项同对象的连续重复表达，原始输出保留。")
                st.text(op["evidence"])
                st.caption(f"意图：{op['intent']} · 原文位置 [{op['start']}, {op['end']})")
                if op.get('scope','current')!='current':
                    st.caption('归属：'+{'other':'其他对象','past':'过去体验','uncertain':'待核验'}[op['scope']]+'；不计入当前对象主题。')
            st.caption("引用可定位；尚不表示语义正确或已经人工审核。")
        if result and 'discarded' in result:
            with st.expander('第二遍机器校对依据与剔除项（不是人工审核）'):
                st.json({'保留意见依据':[{'引用':o['evidence'],'机器依据':o.get('semantic_reason',''),'规则调整':o.get('semantic_adjustments',[])} for o in opinions],
                         '剔除候选':result['discarded']})
        prediction = classifications.get(rid)
        if prediction:
            with st.expander("方面分类器结果"):
                st.dataframe([{"方面": TAXONOMY["aspects"][a], "情感": SENTIMENTS[v], "模型概率": round(c, 3)} for a, v, c in zip(ASPECTS, prediction["labels"], prediction["confidence"]) if v != -2], hide_index=True, width="stretch")
                st.caption("概率未经校准，不代表准确率；分类不提供证据跨度。")
        if rid in reviews:
            st.success(f"已有审核：{reviews[rid]['payload']['decision']} · {reviews[rid]['actor']}")
            with st.expander("查看已保存的审核内容"):
                for op in reviews[rid]["payload"]["opinions"]:
                    st.text(f"{OPINION_LABELS[op['aspect']]} / {SENTIMENTS[op['sentiment']]}：{op['evidence']}")
                    if "target" in op:
                        st.text(f"对象：{op['target'] or '待定'}；意见：{op['issue']}")
                st.text(reviews[rid]["payload"]["note"] or "无补充说明")


if page == "分析概览":
    st.title("听见反馈，追溯原文")
    st.caption(dataset_label(metadata) + " · 当前统计范围为整个数据集")
    if not candidate:
        st.caption('当前展示原始模型基线结果，不是微调模型成绩。可在侧栏创建独立微调分析版本。')
    columns = st.columns(4)
    for col, title, value in zip(columns, ["反馈总数", "已处理" if candidate else "已分类", "有可用意见" if contextual else "有效抽取", "已有审核"], [len(records), len(extractions) if candidate else len(classifications), sum(v["valid"] and (bool(v["opinions"]) if contextual else True) for v in extractions.values()), len(reviews)]):
        col.metric(title, f"{value:,}")
    if contextual:
        st.caption(f"已处理 {len(extractions)} 条：{sum(v.get('fully_valid', False) for v in extractions.values())} 条全部通过本轮字段与定位检查；{sum(v.get('partial', False) for v in extractions.values())} 条有部分意见待复核；{sum(not v['valid'] for v in extractions.values())} 条抽取失败；{sum(v['valid'] and not v['opinions'] for v in extractions.values())} 条返回有效空结果。以上均不代表语义准确率。")
    st.info("v0.1暂不提供时间趋势。统计按反馈条数计算，不能解释为受影响用户数。")
    st.subheader("大家在讨论什么")
    counts = pd.DataFrame(opinion_counts(records,extractions) if candidate else aspect_counts(records, classifications))
    if not counts.empty:
        st.bar_chart(counts.set_index("方面")[["负面", "中性", "正面"]+(['混合'] if candidate else [])],color=['#cc6d64','#99a7af','#40968d']+(['#a889bb'] if candidate else []))
        st.dataframe(counts.drop(columns="aspect"), hide_index=True, width="stretch")
    st.caption('统计来自候选模型通过机械检查的当前对象意见，按不同反馈计数；失败、过去体验及其他对象不计入。同条反馈可有多个情感，合计不等于反馈总数。' if candidate else "数据来自固定 18 方面分类器；同一条反馈可提及多个方面，各方面合计不等于反馈总数。修订抽取的食材/做法扩展标签不改变这份分类统计。")
    with st.container(border=True):
        st.subheader('分析任务')
        done = progress.get("completed", 0)
        if progress.get('phase') == 'semantic_review':
            st.progress(min(progress.get('completed_candidates',0)/max(progress.get('total_candidates',1),1),1),text=f"上下文校对 {progress.get('completed_candidates',0)}/{progress.get('total_candidates',0)} 个候选 · 完成后写入反馈结果")
        elif progress.get("phase") == "context_windows":
            st.progress(min(progress.get("completed_windows",0)/max(progress.get("total_windows",1),1),1),text=f"上下文窗口 {progress.get('completed_windows',0)}/{progress.get('total_windows',0)} · 合并后写入反馈结果")
        else:
            st.progress(min(done / max(len(records), 1), 1), text=f"意见抽取 {done}/{len(records)} · {progress['status']}")
        if progress.get("status") == "failed":
            st.error(progress.get("error", "后台任务失败，可查看日志并重试。"))
        busy = any(job_running(dataset_path(m["dataset_id"])) for m in all_options)
        complete = (folder / "topics.json").exists() and len(extractions) == len(records)
        missing = [] if complete else missing_analysis_assets(folder)
        if complete:
            st.success('分析已完成，可查看主题、复核意见或导出报告。')
        elif busy:
            st.info('已有本地分析任务正在运行，完成后可继续启动。')
        if missing:
            st.warning('分析资源尚未就绪；仍可导入数据和浏览已有结果。')
            with st.expander('查看缺少的资源'):
                st.code('\n'.join(missing),language=None)
                st.caption('启动与资源说明：docs/running-workbench.md。此处不会自动下载模型。')
        if st.button("开始 / 继续本地分析", disabled=busy or complete or bool(missing), type="primary"):
            try:
                start_analysis(folder)
                st.rerun()
            except (ValueError,OSError) as exc:
                st.error(f'未能启动分析：{exc}')
        st.button("刷新进度")
    with st.expander('数据质量详情'):
        st.json(metadata['quality'])

elif page == "反馈与证据":
    st.title("反馈与证据")
    a, b, c = st.columns([2, 1, 1])
    query = a.text_input("搜索原文或记录编号", placeholder="例如：等菜、优惠券、5365")
    aspect = b.selectbox("筛选方面", ["全部"] + (list(OPINION_LABELS) if candidate else ASPECTS), format_func=lambda x: OPINION_LABELS.get(x, x))
    sentiment = c.selectbox("筛选情感", ["全部", "负面", "中性", "正面"]+(['混合','不确定'] if candidate else []))
    filtered = []
    for row in records:
        if query and query != str(row["source_record_id"]) and query.casefold() not in row["text_raw"].casefold():
            continue
        if candidate:
            result = extractions.get(row['record_id'],{})
            ops = result.get('opinions',[]) if result.get('valid') else []
            if (aspect!='全部' or sentiment!='全部') and not any(
                (aspect=='全部' or o['aspect']==aspect) and (sentiment=='全部' or SENTIMENTS[o['sentiment']]==sentiment) for o in ops):
                continue
            filtered.append(row)
            continue
        labels = classifications.get(row["record_id"], {}).get("labels", [])
        selected_labels = [labels[ASPECTS.index(aspect)]] if labels and aspect != "全部" else labels
        if aspect != "全部" and (not selected_labels or selected_labels[0] == -2):
            continue
        if sentiment != "全部" and not any(SENTIMENTS[v] == sentiment for v in selected_labels):
            continue
        filtered.append(row)
    st.caption(f"筛选命中 {len(filtered)} / {len(records)} 条；"+('依据候选模型有效意见，包含过去及其他对象，归属见卡片。' if candidate else '方面和情感筛选依据分类器预测。'))
    if not filtered:
        st.info("没有匹配的反馈，请调整筛选条件。")
    else:
        pages = (len(filtered)+19)//20
        if st.session_state.get("record_page", 1) > pages:
            st.session_state["record_page"] = 1
        number = st.number_input("页码", 1, pages, key="record_page")
        subset = filtered[(number-1)*20:number*20]
        st.dataframe([{"记录": r["source_record_id"], "评分": r["rating"], "原文摘要": r["text_raw"][:90]} for r in subset], hide_index=True, width="stretch")
        rid = st.selectbox("查看反馈", [r["record_id"] for r in subset], format_func=lambda x: by_id[x]["source_record_id"])
        show_record(by_id[rid])
        st.download_button("导出当前筛选的证据报告", report_markdown(folder, filtered, classifications, extractions), file_name="echolens-filtered-report.md", mime="text/markdown")

elif page == "主题候选":
    st.title("主题候选")
    st.caption("从原文意见片段发现相似讨论；主题名称由模型建议，失败时回退为关键词，等待你确认。")
    if not state["topics"]:
        st.info("本轮主题分析已完成，没有形成主题簇；可回查下方离群反馈。" if (folder / "topics.json").exists() else "尚无主题候选。需先完成意见抽取和主题分析；未成组的意见不会被强行分配。")
    else:
        st.caption(f"主题版本 {state['version']} · {len(state['units'])} 个有效意见 · {sum(str(t)=='-1' for t in state['assignments'].values())} 个离群意见")
        topic_map = {str(t["id"]): t for t in state["topics"]}
        available_topics = list(topic_map)
        topic_support = {tid:len({u['record_id'] for u in state['units'] if str(state['assignments'].get(u['unit_id']))==tid}) for tid in topic_map}
        if contextual:
            topic_aspects = {tid:{u["aspect"] for u in state["units"] if str(state["assignments"].get(u["unit_id"])) == tid} for tid in topic_map}
            aspect_choices = sorted(set().union(*topic_aspects.values()))
            topic_aspect = st.selectbox("主题方面", ["全部"]+aspect_choices, format_func=lambda x:OPINION_LABELS.get(x,x))
            available_topics = [tid for tid in topic_map if topic_aspect == "全部" or topic_aspect in topic_aspects[tid]]
            st.caption(f"共 {len(topic_map)} 个主题，本次显示 {len(available_topics)} 个。机器先按预测方面分组，再寻找相似意见；方面判断仍需核验。")
        available_topics.sort(key=lambda tid:-topic_support[tid])
        if metadata.get('analysis_config') == 'configs/stage2-v3.json':
            st.caption('第三轮主主题至少涉及 10 条不同反馈，按反馈数排序；小众意见仍保留在未成组列表中。人工调整后的主题规模可能改变。')
        chosen = st.selectbox("选择主题", available_topics, format_func=lambda x: f"{topic_map[x]['name']} · {topic_support[x]} 条反馈 · #{x}")
        st.caption("关键词：" + " / ".join(topic_map[chosen].get("keywords", [])))
        units = [u for u in state["units"] if str(state["assignments"].get(u["unit_id"])) == chosen]
        st.caption(f"本主题包含 {len(units)} / {len(state['units'])} 个有效意见。")
        if len(units) > 0.5 * len(state["units"]):
            st.warning("此主题覆盖超过半数有效意见，可能过于宽泛，请结合代表片段和边界片段检查。")
        st.metric("涉及反馈", len({u["record_id"] for u in units}), help="同一条反馈在当前主题只计一次")
        left, right = st.columns(2)
        for col, title, subset in [(left, "代表片段", representative_units(units)), (right, "同主题边界片段（不等于负面）", representative_units(units, boundary=True))]:
            with col:
                st.subheader(title)
                displayed = set()
                for unit in subset:
                    evidence_key = (unit["record_id"], unit["start"], unit["end"])
                    if evidence_key in displayed:
                        continue
                    displayed.add(evidence_key)
                    if "target" in unit:
                        st.caption(f"评价对象：{unit['target'] or '待定'}")
                    st.text(unit["evidence"])
                    st.caption(f"记录 {by_id[unit['record_id']]['source_record_id']} · {OPINION_LABELS[unit['aspect']]} · {SENTIMENTS[unit['sentiment']]}")
        st.caption("代表与边界由原始聚类中心距离排序；人工合并/拆分后保留原始距离，不能把边界片段当作已验证反例。")
        inspect_id = st.selectbox("回查主题中的反馈", list(dict.fromkeys(u["record_id"] for u in units)), format_func=lambda x: by_id[x]["source_record_id"])
        with st.expander("查看完整原文和证据"):
            show_record(by_id[inspect_id])
        with st.expander("调整主题（保留历史版本）"):
            operation = st.selectbox("操作", ["重命名", "合并", "拆分"])
            name = st.text_input("调整后的主题名称", value=topic_map[chosen]["name"])
            actor = st.text_input("操作人", key="topic_actor")
            merge_ids = st.multiselect("参与合并的主题", list(topic_map), default=[chosen], format_func=lambda x: f"{topic_map[x]['name']} · #{x}") if operation == "合并" else [chosen]
            unit_ids = st.multiselect("移入新主题的意见", [u["unit_id"] for u in units], format_func=lambda x: next(u["evidence"][:60] for u in units if u["unit_id"] == x)) if operation == "拆分" else []
            if st.button("保存主题调整"):
                try:
                    edit_topics(folder, {"重命名": "rename", "合并": "merge", "拆分": "split"}[operation], merge_ids, name, unit_ids, actor)
                    st.success("主题新版本已保存；再次进入本页可查看。")
                except ValueError as exc:
                    st.error(str(exc))
    outlier_ids = list(dict.fromkeys(u["record_id"] for u in state["units"] if str(state["assignments"].get(u["unit_id"])) == "-1"))
    if outlier_ids:
        with st.expander("回查未成组的离群反馈"):
            outlier_id = st.selectbox("离群反馈", outlier_ids, format_func=lambda x: by_id[x]["source_record_id"])
            show_record(by_id[outlier_id])

elif page == "人工复核":
    st.title("人工复核")
    st.caption("人工决定单独保存，保留原始模型输出；导出时包含审核人和变更历史。")
    reason = st.selectbox("复核范围", ["全部待审核", "抽取失败", "空结果需检查遗漏", "部分意见失败", "方面或对象待定", "低把握", "模型分歧", "随机抽查", "已有审核"])
    queue = [r for r in records if ((r["record_id"] in reviews) if reason == "已有审核" else r["record_id"] not in reviews and (reason == "全部待审核" or reason in review_reasons(r["record_id"], classifications, extractions)))]
    st.caption(f"当前队列 {len(queue)} 条；低把握阈值 0.65，仅用于排序复核，不是质量保证。")
    if queue:
        rid = st.selectbox("待复核反馈", [r["record_id"] for r in queue], format_func=lambda x: by_id[x]["source_record_id"])
        row = by_id[rid]
        st.caption("入队原因：" + "、".join(review_reasons(rid, classifications, extractions)))
        show_record(row)
        result = extractions.get(rid, {"opinions": []})
        prior = reviews.get(rid, {}).get("payload", {}).get("opinions", result["opinions"])
        spans = clauses(row["text_raw"]) if contextual else sentences(row["text_raw"])
        items = [{"方面": op["aspect"], "情感": op["sentiment"], "意图": op["intent"], "原文片段编号": next((s["id"] for s in spans if s["start"] == op["start"] and s["end"] == op["end"]), -1)} for op in prior]
        if contextual:
            scope_labels = {"current":"当前餐厅", "other":"其他对象", "uncertain":"待核验"}
            if candidate:
                scope_labels['past'] = '过去体验'
                st.caption('情感编码：-1负面、0中性、1正面、2混合、3不确定。对象可由完整上下文确定，不要求名称逐字出现在短引文里。')
            items = [{"方面":op["aspect"], "情感":op["sentiment"], "意图":op["intent"],
                      "起始字符":op["start"], "结束字符":op["end"], "归属":scope_labels[op.get("scope","current")],
                      "评价对象":op["target"], "具体意见":op["issue"]} for op in prior]
            st.caption("字符位置从 0 开始，结束位置不包含在引用中；不修改位置就保留原引用。归属不明确时选择待核验。审核不会自动重算机器主题。")
        with st.form("review_"+rid):
            columns = ["方面", "情感", "意图", "起始字符", "结束字符", "归属", "评价对象", "具体意见"] if contextual else ["方面", "情感", "意图", "原文片段编号"]
            frame = pd.DataFrame(items, columns=columns)
            edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, width="stretch", column_config={
                "方面": st.column_config.SelectboxColumn(options=list(OPINION_LABELS if candidate else LEGACY_OPINION_LABELS) if contextual else ASPECTS, required=True),
                "情感": st.column_config.SelectboxColumn(options=[-1, 0, 1]+([2,3] if candidate else []), required=True),
                "意图": st.column_config.SelectboxColumn(options=TAXONOMY["intents"], required=True),
                "原文片段编号": st.column_config.NumberColumn(min_value=0, max_value=max(len(spans)-1, 0), step=1, required=True),
                "归属": st.column_config.SelectboxColumn(options=["当前餐厅","其他对象","待核验"]+(['过去体验'] if candidate else []), required=True),
                "起始字符": st.column_config.NumberColumn(min_value=0, max_value=len(row["text_raw"]), step=1, required=True),
                "结束字符": st.column_config.NumberColumn(min_value=0, max_value=len(row["text_raw"]), step=1, required=True)})
            decision = st.selectbox("审核结论", ["确认可用", "已修正", "不适用 / 拒绝"])
            note = st.text_area("审核说明")
            actor = st.text_input("审核者")
            submitted = st.form_submit_button("保存审核", type="primary")
        if submitted:
            try:
                if contextual:
                    opinions = [{"aspect":r["方面"], "sentiment":int(r["情感"]), "intent":r["意图"],
                                 "start":int(r["起始字符"]), "end":int(r["结束字符"]), "scope":{v:k for k,v in scope_labels.items()}.get(r["归属"]),
                                 "target":r["评价对象"] or "", "issue":r["具体意见"],
                                 "evidence":row["text_raw"][int(r["起始字符"]):int(r["结束字符"])]} for r in edited.to_dict("records")]
                else:
                    raw = {"opinions": [{"a": ASPECTS.index(r["方面"]), "s": int(r["情感"]), "i": r["意图"], "e": int(r["原文片段编号"])} for r in edited.to_dict("records")]}
                    opinions = decode_opinions(json.dumps(raw, ensure_ascii=False), row["text_raw"], max_opinions=100)
                save_review(folder, rid, opinions, {"确认可用": "approved", "已修正": "edited", "不适用 / 拒绝": "rejected"}[decision], note, actor)
                st.success("审核记录已保存，机器原始结果保留不变。")
            except (ValueError, TypeError) as exc:
                st.error(f"无法保存：{exc}")
    else:
        st.info("当前范围没有待处理记录。")
elif page == '报告导出':
    st.title('报告导出')
    st.caption(dataset_label(metadata)+' · 导出整个数据集；筛选报告请在“主题与证据”的原文检索中导出。')
    st.info('报告区分机器统计和人工修订；未通过检查的机器输出不作为可靠证据。')
    st.download_button("下载带证据的 Markdown 报告", report_markdown(folder, records, classifications, extractions), file_name="echolens-report.md", mime="text/markdown")
    history = events(folder)
    st.download_button("下载审核与主题变更 JSONL", "".join(json.dumps(e, ensure_ascii=False)+"\n" for e in history), file_name="echolens-review-events.jsonl", mime="application/x-ndjson")
    st.caption(f"已保存 {len(history)} 条变更记录。导出包含原文片段，仅供本地审核或在许可范围内分享。")
