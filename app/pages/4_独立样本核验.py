import json
from collections import Counter

import streamlit as st
from echolens.independent_review import PACKET, RUN, ISSUES, CHOICES, load_packet, predictions, view, history, save_review, digest
from echolens.stage3_sft import sha256
from echolens.workbench import highlighted
from echolens.annotation import SENTIMENTS, INTENTS, SCOPES

st.set_page_config(page_title='EchoLens · 独立样本核验',layout='wide',initial_sidebar_state='collapsed')
st.markdown('<a href="/" target="_self">← 返回餐饮工作台</a>',unsafe_allow_html=True)
st.markdown('<style>.block-container{padding:1.5rem;max-width:1800px}.evidence-text{white-space:pre-wrap;line-height:1.9;padding:16px;background:white;border:1px solid #dfe5e8;border-radius:8px;overflow-wrap:anywhere}mark{background:#ffefb5;color:#25363f}</style>',unsafe_allow_html=True)
st.title('独立样本核验 · 微调前后对照')
st.caption('50条未用于本项目训练和调参的测试分区评论。先读原文找意见，再比较两版；不需要重写JSON。此处保存的是人工成对审核，不是完整金标准标注。')
try:
    manifest, rows, aspects = load_packet(PACKET)
except ValueError as exc:
    st.info(str(exc))
    st.stop()
before, after = predictions(RUN/'before'), predictions(RUN/'after')
events = history(PACKET)
saved = {e['record_id']:e for e in events}
completed = sum(e['status']=='completed' for e in saved.values())
st.progress(completed/len(rows),text=f'已审核 {completed}/{len(rows)} · 已生成：微调前 {len(before)}/50，微调后 {len(after)}/50')
acceptance_path = PACKET/'acceptance.json'
if acceptance_path.exists():
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    if acceptance.get('packet_sha256')==sha256(PACKET/'manifest.json') and acceptance.get('accepted'):
        st.success('用户已在对话中确认整批核验通过。上方进度仅统计实际保存的逐条审核表，不代填记录，不据此计算人工准确率。')
with st.expander('怎么审：重点看这六类问题',expanded=False):
    st.markdown('''1. **漏提**：先从完整原文找出意见，尤其后半段、子对象、中性事实；不要只看高亮。
2. **对象和范围**：某道菜、配料、同行或过去体验，是否被扩大成整店评价？
3. **证据边界**：否定、转折和条件是否齐全？是否混入另一对象或相反态度？
4. **方面与情感**：食材搭配不能误当分量；规则说明不能自动当服务差。含蓄、讽刺请保留疑问。
5. **重复与合并**：同义意见是否重复，不同对象是否错误合并？一句可以有多条意见。
6. **退步和失败**：微调后是否丢掉原来正确的意见？格式失败、超长输出同样要计入。

只需勾选每版问题并写必要备注，例如“微调后#3把配料当分量；原文后半段漏了口感”。前10条短中长混排，可先审核这10条再继续。''')

with st.sidebar:
    st.subheader('审核进度')
    scope = st.radio('浏览范围',['全部50条','先看前10条','仅未完成'])
    actor = st.text_input('审核者',value='用户')
    st.download_button('导出全部审核历史',json.dumps({'packet_sha256':sha256(PACKET/'manifest.json'),'events':events},ensure_ascii=False,indent=2),file_name='independent50-reviews.json',mime='application/json')
    counts = Counter(e['choice'] for e in saved.values() if e['status']=='completed')
    if counts:
        st.write({CHOICES[k]:v for k,v in counts.items()})
    st.caption('只统计你已保存的判断；未审核不会自动计为通过。审核结果不进入训练。')
options = [i for i,r in enumerate(rows) if scope=='全部50条' or (scope=='先看前10条' and i<10) or (scope=='仅未完成' and saved.get(r['record_id'],{}).get('status')!='completed')]
if not options:
    st.success('当前范围均已完成；可以切换范围或导出审核记录。')
    st.stop()
if 'independent_next' in st.session_state:
    st.session_state['independent_index'] = st.session_state.pop('independent_next')
if st.session_state.get('independent_index') not in options:
    st.session_state['independent_index'] = options[0]
index = st.selectbox('选择评论',options,key='independent_index',format_func=lambda i:f"{i+1:02d}/50 · 记录 {rows[i]['source_record_id']} · {len(rows[i]['text'])}字 · {'已审核' if saved.get(rows[i]['record_id'],{}).get('status')=='completed' else '待审核'}")
row = rows[index]
rid = row['record_id']
pair = {'before':before.get(rid),'after':after.get(rid)}
views = {side:view(result,row['text'],aspects) for side,result in pair.items()}
if 'independent_notice' in st.session_state:
    st.success(st.session_state.pop('independent_notice'))
st.subheader(f"原文 · {row['source_record_id']}")
show = st.checkbox('显示两版结果',value=False,key=f'show_{rid}',help='建议先读完整原文，自己找出主要意见，再打开对照，减少被模型高亮带偏。')
with st.container():
    highlight = st.selectbox('原文高亮',['不高亮','微调前全部证据','微调后全部证据'],key=f'highlight_{rid}') if show else '不高亮'
    spans = []
    if highlight!='不高亮':
        selected_view = views['before' if highlight=='微调前全部证据' else 'after']
        located = selected_view.get('located',{})
        focus = st.selectbox('定位具体意见',[0]+list(located),format_func=lambda n:'全部可定位证据' if n==0 else f'意见 #{n}',key=f'focus_{rid}_{highlight}')
        spans = list(located.values()) if focus==0 else [located[focus]]
    with st.container(height=300):
        st.markdown(highlighted(row['text'],spans),unsafe_allow_html=True)
    st.caption('高亮仅表示证据能定位；未高亮处仍可能有漏提。可以选择某条意见单独高亮。')
middle, right = st.columns(2)
for column, side, title in [(middle,'before','微调前'),(right,'after','微调后 · 第102步')]:
    with column:
        st.subheader(title)
        v = views[side]
        if not show:
            st.info('读完原文后勾选“显示两版结果”。')
            continue
        if not v['ready']:
            st.info('结果尚未生成，请稍后刷新。')
            continue
        if v['hit_limit']:
            st.error('达到输出长度上限，可能不完整；请作为失败情况审核。')
        if not v['checked']['json_valid']:
            st.error('结果不是合规JSON，不能将未显示的意见当作“没有意见”。')
        elif not v['checked']['schema_valid']:
            st.warning('含无效字段或标签；下面保留原始意见，不静默删除。')
        st.caption(f"{len(v['ops'])}条输出意见 · {v['seconds']:.1f}秒 · 定位通过不等于语义正确")
        if v['checked']['json_valid'] and not v['ops']:
            st.warning('模型输出空意见；请检查原文是否确实没有相关评价。')
        for n,op in enumerate(v['ops'],1):
            with st.container(border=True):
                if not isinstance(op,dict):
                    st.text(f'#{n} 无效意见：{op}')
                    continue
                st.text(f"#{n} · {op.get('target','未提供对象')}\n{aspects.get(str(op.get('aspect')),op.get('aspect','未提供方面'))} · {SENTIMENTS.get(str(op.get('sentiment')),op.get('sentiment','未提供情感'))}")
                st.text(op.get('evidence','未提供证据'))
                if n not in v['located']:
                    st.warning('这条证据无法按原文定位，请复核。')
                st.caption(f"意图：{INTENTS.get(str(op.get('intent')),op.get('intent'))} · 归属：{SCOPES.get(str(op.get('scope')),op.get('scope'))}")
        with st.expander('原始输出（含格式失败文本）'):
            st.code(v['raw'],language='json')

st.divider()
st.subheader('你的审核')
current = saved.get(rid,{})
revision_key = f'independent_revision_{rid}'
st.session_state.setdefault(revision_key,current.get('revision',0))
with st.form(f'review_{rid}'):
    a,b = st.columns(2)
    before_issues = a.multiselect('微调前的问题',ISSUES,default=current.get('before_issues',[]))
    after_issues = b.multiselect('微调后的问题',ISSUES,default=current.get('after_issues',[]))
    choices = list(CHOICES)
    choice = st.selectbox('总体判断',choices,index=choices.index(current.get('choice','unreviewed')),format_func=CHOICES.get)
    note = st.text_area('问题备注（建议写版本、意见编号、遗漏原文或原因）',value=current.get('note',''))
    coverage = st.checkbox('我已读完整条原文，并检查两版的遗漏、对象、证据和情感',value=current.get('coverage',False))
    ready = all(v['ready'] for v in views.values()) and show
    x,y = st.columns(2)
    draft = x.form_submit_button('保存草稿',disabled=not ready)
    finish = y.form_submit_button('保存完成并下一条',disabled=not ready,type='primary')
    st.caption('不需要重写模型JSON。保存完成只代表你审核了这组对照，不会生成完整金标准，也不会用于调参。切换评论或关闭前请保存。')
    if draft or finish:
        try:
            result = save_review(PACKET,rid,sha256(PACKET/'manifest.json'),digest(pair),
                {'actor':actor,'choice':choice,'before_issues':before_issues,'after_issues':after_issues,
                 'note':note,'coverage':coverage,'status':'completed' if finish else 'draft'},
                 expected_revision=st.session_state[revision_key])
            st.session_state[revision_key] = result['revision']
            st.session_state['independent_notice'] = '审核已保存。' if finish else '草稿已保存。'
            if finish and options.index(index)+1<len(options):
                st.session_state['independent_next'] = options[options.index(index)+1]
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
