"""Plot support view."""

import streamlit as st

from ..llm_services import ensure_entry_summaries, generate_plot_script, should_use_llm
from ..models import ProjectData, get_entry_slice
from ..utils import create_docx_bytes


def render(project: ProjectData, client) -> None:
    st.header("プロット支援")
    llm_mode = should_use_llm(project, client)
    ensure_entry_summaries(project, client if llm_mode else None)
    if llm_mode:
        st.caption("指定区間を元に会話形式の叩き台を生成します。Speakers ラベルは本文に基づきます。")
    else:
        st.caption("このプロジェクトではサンプルのプロット叩き台を提示します。Speakers ラベルはサンプル想定です。")

    if project.chunk_count == 0:
        st.warning("チャンクデータがありません。")
        return

    col_start, col_end = st.columns(2)
    with col_start:
        start_idx = st.number_input("開始チャンク", min_value=1, max_value=project.chunk_count, value=1)
    with col_end:
        end_idx = st.number_input(
            "終了チャンク",
            min_value=start_idx,
            max_value=project.chunk_count,
            value=min(project.chunk_count, start_idx + 4),
        )

    entries = get_entry_slice(project, start_idx, end_idx)
    st.markdown(f"選択チャンク: {entries[0].id if entries else 'なし'} 〜 {entries[-1].id if entries else 'なし'}")

    plot_state_key = f"latest_plot_script_{project.key}"
    edit_key = f"plot_editor_{project.key}"
    if st.button("プロット叩き台を生成", type="primary"):
        script = generate_plot_script(client, project, entries, project.characters)
        st.session_state[plot_state_key] = script
        st.session_state[edit_key] = script

    script_text = st.session_state.get(plot_state_key)
    if script_text is None:
        return

    st.subheader("生成結果")
    st.session_state.setdefault(edit_key, script_text)

    col_script, col_source = st.columns([3, 2])
    with col_script:
        st.text_area("シナリオ案（編集可）", key=edit_key, height=420)
        edited_script = st.session_state.get(edit_key, "")
        docx_bytes = create_docx_bytes(edited_script)
        if docx_bytes:
            st.download_button(
                "Wordファイルとしてダウンロード",
                data=docx_bytes,
                file_name=f"{project.key}_plot.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    with col_source:
        st.text_area("本文（リファレンス）", value=project.full_text, height=420, disabled=True)
