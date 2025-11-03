"""Project management view for Streamlit UI."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from .. import config
from ..llm_services import build_character_glossary, ensure_entry_summaries
from ..llm_workflow import llm_cut_and_label_with_params
from ..models import EntryRecord, ProjectData, panels_to_entries
from ..storage import (
    delete_project_files,
    load_project_definitions,
    register_project,
    remove_project_definition,
    save_character_file,
    save_project_definition,
    save_project_payload,
)
from . import add_project

logger = logging.getLogger(__name__)


def render(client) -> None:
    st.header("プロジェクトの管理")
    st.caption("新規プロジェクトの追加や既存プロジェクトの編集・削除、再ラベル付けを行います。")

    add_project.render(client, show_header=False)
    st.divider()

    definitions: List[Dict[str, object]] = st.session_state.get("project_definitions", [])
    if not definitions:
        definitions = load_project_definitions()
        st.session_state["project_definitions"] = definitions

    projects: Dict[str, ProjectData] = st.session_state.get("projects", {})
    if not projects:
        st.info("登録済みのプロジェクトがありません。先にプロジェクトを追加してください。")
        return

    st.subheader("登録済みプロジェクト")
    for definition in definitions:
        key = definition.get("key")
        if not key:
            continue
        project = projects.get(key)
        if project is None:
            continue
        expanded = key == st.session_state.get("current_project")
        label = f"{project.title}（{key}）"
        with st.expander(label, expanded=expanded):
            _render_project_controls(project, definition, client)


def _render_project_controls(
    project: ProjectData,
    definition: Dict[str, object],
    client,
) -> None:
    project_dir, panel_path, characters_path = _resolve_storage_paths(project)

    with st.form(f"meta-form-{project.key}"):
        new_title = st.text_input("プロジェクト名", value=project.title)
        summary_value = st.text_area("作品全体サマリー", value=project.summary, height=160)
        save_meta = st.form_submit_button("基本情報を保存", use_container_width=True)

    if save_meta:
        project.title = new_title.strip() or project.title
        project.summary = summary_value.strip()
        project.source_path = panel_path
        _persist_project(project, panel_path, characters_path)
        _update_definition(project, panel_path, characters_path)
        st.success("基本情報を保存しました。")

    st.markdown("#### 登場人物一覧")
    with st.form(f"characters-form-{project.key}"):
        editor_value = st.data_editor(
            project.characters or [],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", required=True),
                "Role": st.column_config.TextColumn("Role"),
                "Details": st.column_config.TextColumn("Details", max_chars=400),
            },
            key=f"characters-editor-{project.key}",
        )
        save_characters = st.form_submit_button("登場人物を保存", use_container_width=True)

    if save_characters:
        characters = _normalise_characters(editor_value)
        project.characters = characters
        save_character_file(characters_path, characters)
        _persist_project(project, panel_path, characters_path)
        st.success("登場人物を更新しました。")

    st.markdown("#### 再ラベル付け")
    st.caption("登場人物の更新後などに全文を再分割・ラベル付けします。OpenAI API キーが必要です。")
    chunk_target = st.slider(
        "再ラベル時の目標カット長（文字数）",
        min_value=80,
        max_value=600,
        value=180,
        step=10,
        key=f"chunk-target-{project.key}",
    )
    relabel_col1, relabel_col2 = st.columns([2, 1])
    with relabel_col1:
        st.caption(f"保存先: {panel_path}")
    can_relabel = bool(project.full_text.strip())
    if not can_relabel:
        st.info("このプロジェクトには全文が保存されていません。再ラベル付けを行うには、プロジェクト追加時に全文を取り込んでください。")

    with relabel_col2:
        relabel_clicked = st.button(
            "LLMで再ラベル付けを実行",
            key=f"relabel-btn-{project.key}",
            disabled=not can_relabel,
            type="primary",
        )

    if relabel_clicked:
        if client is None:
            st.error("OpenAI API キーが設定されていないため、再ラベル付けを実行できません。")
        else:
            _run_relabel_pipeline(project, panel_path, characters_path, chunk_target, client)

    st.markdown("#### プロジェクトの削除")
    if project.key in config.SAMPLE_ONLY_PROJECT_KEYS:
        st.caption("サンプルプロジェクトは削除できません。")
    else:
        if st.button(f"プロジェクト「{project.title}」を削除", key=f"delete-{project.key}", type="secondary"):
            st.session_state["manage_delete_candidate"] = project.key
        if st.session_state.get("manage_delete_candidate") == project.key:
            st.warning("この操作は元に戻せません。関連ファイルも削除されます。")
            confirm = st.button(
                "削除を確定",
                key=f"delete-confirm-{project.key}",
                type="primary",
            )
            if confirm:
                _delete_project(project.key, panel_path, characters_path)
                st.success(f"プロジェクト「{project.title}」を削除しました。")
                st.session_state.pop("manage_delete_candidate", None)
                st.rerun()


def _resolve_storage_paths(project: ProjectData) -> Tuple[Path, Path, Path]:
    project_dir = (config.DATA_DIR / project.key).resolve()
    panel_path = project.source_path.resolve() if project.source_path else project_dir / "project.json"
    characters_path = project_dir / "characters.json"

    if not _is_under(panel_path, project_dir):
        panel_path = project_dir / "project.json"
    if not _is_under(characters_path, project_dir):
        characters_path = project_dir / "characters.json"
    return project_dir, panel_path, characters_path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _normalise_characters(value) -> List[Dict[str, str]]:
    if value is None:
        return []
    records: List[Dict[str, str]] = []
    if isinstance(value, list):
        records = [dict(item) for item in value]
    else:
        try:
            records = value.to_dict(orient="records")  # type: ignore[attr-defined]
        except AttributeError:
            return []
    normalised: List[Dict[str, str]] = []
    for row in records:
        name = str(row.get("Name") or "").strip()
        role = str(row.get("Role") or "").strip()
        details = str(row.get("Details") or "").strip()
        if not (name or role or details):
            continue
        if not name:
            continue
        normalised.append({"Name": name, "Role": role, "Details": details})
    return normalised


def _persist_project(project: ProjectData, panel_path: Path, characters_path: Path) -> None:
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    project.source_path = panel_path
    save_project_payload(panel_path, project)
    save_character_file(characters_path, project.characters)
    _update_definition(project, panel_path, characters_path)


def _update_definition(project: ProjectData, panel_path: Path, characters_path: Path) -> None:
    definition_payload = {
        "key": project.key,
        "title": project.title,
        "panel_file": panel_path,
        "character_file": characters_path,
    }
    save_project_definition(definition_payload)
    register_project(definition_payload, project)


def _run_relabel_pipeline(
    project: ProjectData,
    panel_path: Path,
    characters_path: Path,
    chunk_target: int,
    client,
) -> None:
    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    st.session_state["progress_text"] = progress_text
    st.session_state["progress_bar"] = progress_bar
    glossary = build_character_glossary(project.characters)
    canonical_names = [c.get("Name", "") for c in project.characters]

    try:
        progress_text.text("本文を再分割しています…")
        panels = llm_cut_and_label_with_params(
            client,
            project.full_text,
            character_glossary=glossary,
            chunk_target=int(chunk_target),
        )
        if not panels:
            st.error("再ラベル付けに失敗しました。出力が空です。")
            return
        entries: List[EntryRecord] = panels_to_entries(panels, canonical_names=canonical_names)
        if not entries:
            st.error("エントリーの生成に失敗しました。")
            return
        project.entries = entries
        ensure_entry_summaries(project, client)
        _persist_project(project, panel_path, characters_path)
        st.success("再ラベル付けが完了しました。")
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        logger.exception("再ラベル付けに失敗: %s", exc)
        st.error(f"再ラベル付けでエラーが発生しました: {exc}")
    finally:
        progress_text.empty()
        progress_bar.empty()
        st.session_state.pop("progress_text", None)
        st.session_state.pop("progress_bar", None)


def _delete_project(key: str, panel_path: Path, characters_path: Path) -> None:
    projects = st.session_state.get("projects", {})
    projects.pop(key, None)
    definitions = [d for d in st.session_state.get("project_definitions", []) if d.get("key") != key]
    st.session_state["project_definitions"] = definitions
    remove_project_definition(key)
    delete_project_files(key)

    if st.session_state.get("current_project") == key:
        st.session_state.pop("current_project", None)
