"""Render the original understanding view."""

import streamlit as st

from .. import config
from ..llm_services import (
    emphasize_character_names,
    ensure_entry_summaries,
    should_use_llm,
    summarize_section,
)
from ..models import ProjectData, entries_to_context, get_entry_slice


def render(project: ProjectData, client) -> None:
    st.header("原作理解")
    llm_mode = should_use_llm(project, client)
    ensure_entry_summaries(project, client if llm_mode else None)
    if llm_mode:
        st.caption("OpenAI API を用いて指定区間の要約を生成します。")
    else:
        st.caption("このプロジェクトではサンプル要約を表示します。区間と粒度を指定して体験してください。")
    st.write(f"チャンク総数: {project.chunk_count}")

    if project.summary:
        with st.expander("作品全体の要約", expanded=True):
            st.write(project.summary)

    col_range, col_grain = st.columns([2, 1])
    with col_range:
        range_mode = st.radio("区間を選択", ["全体", "チャンク範囲"], horizontal=True)
        if range_mode == "全体":
            start_idx = 1
            end_idx = project.chunk_count
        else:
            start_idx = st.number_input("開始チャンク（1〜N）", min_value=1, max_value=project.chunk_count, value=1)
            end_idx = st.number_input(
                "終了チャンク（1〜N）",
                min_value=start_idx,
                max_value=project.chunk_count,
                value=min(project.chunk_count, start_idx + 4),
            )
    with col_grain:
        default_grain = 200 if 200 in config.SUMMARY_GRAIN_OPTIONS else config.SUMMARY_GRAIN_OPTIONS[0]
        grain = st.select_slider("粒度（目安文字数）", options=config.SUMMARY_GRAIN_OPTIONS, value=default_grain)

    entries = get_entry_slice(project, start_idx, end_idx)
    st.markdown(f"選択中のチャンク: {entries[0].id if entries else 'なし'} 〜 {entries[-1].id if entries else 'なし'}")

    summary_state_key = f"latest_summary_{project.key}"
    if st.button("要約を生成", type="primary"):
        summary = summarize_section(client, project, entries, grain)
        st.session_state[summary_state_key] = summary

    summary_value = st.session_state.get(summary_state_key)
    if summary_value:
        st.subheader("要約結果")
        formatted = emphasize_character_names(summary_value, project)
        st.markdown(formatted.replace("\n", "  \n"))

    with st.expander("参考：選択チャンクの本文", expanded=False):
        if entries:
            st.write(entries_to_context(entries, config.MAX_CONTEXT_CHARS))
        else:
            st.info("チャンクが選択されていません。")
