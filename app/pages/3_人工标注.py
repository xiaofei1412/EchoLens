import streamlit as st
from echolens.annotation_ui import render_annotation_page

st.set_page_config(page_title="EchoLens · 人工试标", layout="wide", initial_sidebar_state="collapsed")
st.markdown('<a href="/" target="_self">← 返回餐饮工作台</a>',unsafe_allow_html=True)
render_annotation_page()
