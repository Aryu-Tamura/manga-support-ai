"""Character analysis view."""

import streamlit as st

from ..llm_services import ensure_entry_summaries, generate_character_analysis, should_use_llm
from ..models import ProjectData, find_character_contexts


def render(project: ProjectData, client) -> None:
    st.header("キャラ解析")
    llm_mode = should_use_llm(project, client)
    ensure_entry_summaries(project, client if llm_mode else None)
    if llm_mode:
        st.caption("OpenAI API を使って選択キャラクターの解析メモを生成します。")
    else:
        st.caption("このプロジェクトではサンプルのキャラ解析結果を表示します。")

    if not project.characters:
        st.warning("キャラクターデータが登録されていません。")
        return

    name_to_character = {c.get("Name", f"キャラ{i}"): c for i, c in enumerate(project.characters, start=1)}
    character_names = list(name_to_character.keys())
    selected_name = st.selectbox("キャラクターを選択", options=character_names)
    character = name_to_character[selected_name]

    st.subheader("キャラクター情報")
    st.markdown(f"**名前**: {character.get('Name', '不明')}")
    st.markdown(f"**役割**: {character.get('Role', '不明')}")
    st.markdown("**詳細メモ**")
    st.write(character.get("Details", "（説明なし）"))

    state_key = f"latest_character_analysis_{project.key}_{selected_name}"
    if st.button("キャラ解析を生成", type="primary"):
        result = generate_character_analysis(client, project, character)
        st.session_state[state_key] = result

    analysis_value = st.session_state.get(state_key)
    if analysis_value:
        st.subheader("解析結果")
        st.write(analysis_value)

    contexts = find_character_contexts(project, character.get("Name", ""))
    with st.expander("参考：本文抜粋", expanded=False):
        if contexts:
            for pid, snippet in contexts:
                st.markdown(f"- **{pid}**: {snippet}")
        else:
            st.info("本文内に該当キャラの記述が見つかりませんでした。")
