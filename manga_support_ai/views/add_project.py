"""UI for adding new projects via file upload and LLM processing."""

import logging

import streamlit as st

from ..config import DATA_DIR
from ..llm_services import (
    build_character_glossary,
    ensure_entry_summaries,
    extract_primary_characters,
    generate_overall_summary,
)
from ..llm_workflow import llm_cut_and_label_with_params
from ..models import EntryRecord, ProjectData, panels_to_entries
from ..storage import (
    generate_project_key,
    register_project,
    save_character_file,
    save_project_definition,
    save_project_payload,
)
from ..utils import read_uploaded_text


def render(client, *, show_header: bool = True) -> None:
    if show_header:
        st.header("プロジェクトを追加")
    else:
        st.subheader("プロジェクトを追加")
    if client is None:
        st.error("OpenAI API キーが設定されていません。プロジェクト追加には API キーが必要です。")
        return

    st.markdown(
        "テキストファイル（.txt）または EPUB をアップロードし、"
        "LLM による分割・ラベル付けと要約を実行して新しいプロジェクトを作成します。"
    )

    existing_keys = set(st.session_state.get("projects", {}).keys())
    for definition in st.session_state.get("project_definitions", []):
        existing_keys.add(definition.get("key"))
    existing_keys = [key for key in existing_keys if key]

    with st.form("add_project_form", clear_on_submit=False):
        title = st.text_input("作品タイトル", "")
        uploaded_file = st.file_uploader("原作ファイル（.txt / .epub）", type=["txt", "text", "epub"])
        chunk_target = st.number_input("目標カット長（文字数）", min_value=80, max_value=600, value=180, step=10)
        style_hint = st.text_area("作風ヒント（任意）", height=80)
        submitted = st.form_submit_button("LLMでプロジェクトを生成", type="primary")

    if not submitted:
        return

    if not title.strip():
        st.warning("作品タイトルを入力してください。")
        return
    if uploaded_file is None:
        st.warning("原作ファイルをアップロードしてください。")
        return

    try:
        full_text = read_uploaded_text(uploaded_file).strip()
    except Exception as exc:
        st.error(f"ファイルの読み込みに失敗しました: {exc}")
        return

    if not full_text:
        st.warning("本文を解析できませんでした。別のファイルでお試しください。")
        return

    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    step_labels = [
        "テキスト読み込み",
        "全体要約生成",
        "登場人物抽出",
        "本文分割＆ラベル付与",
        "チャンク要約生成",
        "ファイル保存＆登録",
        "完了",
    ]

    def update_status(current_index: int, message: str) -> None:
        lines = []
        for idx, label in enumerate(step_labels):
            if idx < current_index:
                prefix = "✅"
            elif idx == current_index:
                prefix = "🔄"
            else:
                prefix = "▫️"
            lines.append(f"{prefix} {label}")
        status_placeholder.markdown("\n".join(lines))
        progress_text.text(message)

    update_status(0, "テキストを解析しています…")

    try:
        update_status(1, "作品要約を生成しています…")
        summary = generate_overall_summary(client, title.strip(), full_text)
        progress_bar.progress(0.2)

        update_status(2, "登場人物を抽出しています…")
        characters = extract_primary_characters(client, full_text)
        progress_bar.progress(0.35)

        glossary = build_character_glossary(characters)

        update_status(3, "本文を分割してラベル付けしています…")
        try:
            panels = llm_cut_and_label_with_params(
                client,
                full_text,
                style_hint=style_hint.strip(),
                character_glossary=glossary,
                chunk_target=int(chunk_target),
            )
            entries = panels_to_entries(
                panels,
                canonical_names=[c.get("Name", "") for c in characters],
            )
        except Exception as exc:  # pragma: no cover - LLM failure fallback
            logging.warning("LLMによる分割に失敗したためフォールバックします: %s", exc)
            entries = [
                EntryRecord(
                    id=1,
                    text=full_text,
                    type="narration",
                    speakers=[],
                    time="unknown",
                    location="",
                    tone="neutral",
                    emotion="neutral",
                    action="",
                    entities=[],
                    source_span={"start": 0, "end": len(full_text)},
                    summary="",
                )
            ]
        if not entries:
            st.error("分割結果が空でした。入力テキストを確認してください。")
            return
        progress_bar.progress(0.7)

        update_status(4, "各チャンクの要約を生成しています…")
        temp_project = ProjectData(
            key="temp",
            title=title.strip(),
            summary=summary,
            entries=entries,
            characters=characters,
            full_text=full_text,
        )
        ensure_entry_summaries(temp_project, client)
        entries = temp_project.entries
        progress_bar.progress(0.85)

        update_status(5, "ファイルを保存しています…")
        progress_bar.progress(0.9)
        key = generate_project_key(title.strip(), existing_keys)
        project_dir = DATA_DIR / key
        panel_file = project_dir / "project.json"
        character_file = project_dir / "characters.json"

        project = ProjectData(
            key=key,
            title=title.strip(),
            summary=summary,
            entries=entries,
            characters=characters,
            full_text=full_text,
            source_path=panel_file,
        )

        save_project_payload(panel_file, project)
        save_character_file(character_file, characters)
        save_project_definition({
            "key": key,
            "title": project.title,
            "panel_file": panel_file,
            "character_file": character_file,
        })
        register_project(
            {
                "key": key,
                "title": project.title,
                "panel_file": panel_file,
                "character_file": character_file,
            },
            project,
        )

        progress_bar.progress(1.0)
        update_status(6, "完了しました。")
        st.session_state["current_view"] = "original"
        st.session_state["project_added_notice"] = project.title
        st.session_state["current_project"] = key
        st.rerun()

    except Exception as exc:  # pragma: no cover - interactive feedback
        logging.exception("プロジェクト追加中にエラー: %s", exc)
        st.error(f"プロジェクトの生成に失敗しました: {exc}")
    finally:
        if progress_bar is not None:
            progress_bar.empty()
        if progress_text is not None:
            progress_text.empty()
        status_placeholder.empty()
