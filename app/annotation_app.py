"""Standalone entry for the same annotation page shown in the workbench."""
import streamlit as st
from echolens.annotation_ui import render_annotation_page

st.set_page_config(page_title="EchoLens · 人工试标", layout="wide", initial_sidebar_state="collapsed")
render_annotation_page()
