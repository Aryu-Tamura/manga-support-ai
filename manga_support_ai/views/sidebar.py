"""Sidebar navigation for the Streamlit UI."""

from typing import List

import streamlit as st

from ..storage import load_project_definitions


def render_sidebar() -> None:
    st.sidebar.markdown("### 📖 プロット作成支援 AI")

    definitions: List[dict] = st.session_state.get("project_definitions", [])
    if not definitions:
        definitions = load_project_definitions()
        st.session_state["project_definitions"] = definitions

    if not definitions:
        st.sidebar.warning("利用可能なプロジェクトがありません。追加してください。")
        if st.sidebar.button("プロジェクトの管理", use_container_width=True):
            st.session_state["current_view"] = "manage"
        return

    project_titles = {item["key"]: item["title"] for item in definitions}
    project_keys = list(project_titles.keys())
    current_key = st.session_state.get("current_project", project_keys[0])
    if current_key not in project_titles:
        current_key = project_keys[0]
        st.session_state["current_project"] = current_key

    selected_key = st.sidebar.selectbox(
        "プロジェクトを選択",
        options=project_keys,
        index=project_keys.index(current_key),
        format_func=lambda key: project_titles[key],
    )
    if selected_key != current_key:
        st.session_state["current_project"] = selected_key

    st.sidebar.divider()
    if st.sidebar.button("原作理解", use_container_width=True):
        st.session_state["current_view"] = "original"
    if st.sidebar.button("キャラ解析", use_container_width=True):
        st.session_state["current_view"] = "character"
    if st.sidebar.button("プロット支援", use_container_width=True):
        st.session_state["current_view"] = "plot"
    if st.sidebar.button("原作理解の検証1", use_container_width=True):
        st.session_state["current_view"] = "validation"

    st.sidebar.divider()
    st.sidebar.markdown("#### 👤 Demo User")
    if st.sidebar.button("プロジェクトの管理", use_container_width=True):
        st.session_state["current_view"] = "manage"
    st.sidebar.caption("プロジェクトの追加・編集・削除を行えます。")
