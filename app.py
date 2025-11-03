"""Streamlit entry-point for the Manga Support AI demo."""

from typing import Optional

import streamlit as st

from manga_support_ai.llm_services import init_client
from manga_support_ai.models import ProjectData
from manga_support_ai.storage import ensure_projects_loaded
from manga_support_ai.utils import setup_logging
from manga_support_ai.views import (
    add_project,
    character,
    manage,
    original,
    plot,
    sidebar,
    validation,
)


def get_current_project() -> Optional[ProjectData]:
    projects = st.session_state.get("projects", {})
    if not projects:
        return None
    current_key = st.session_state.get("current_project")
    if current_key not in projects:
        current_key = next(iter(projects.keys()))
        st.session_state["current_project"] = current_key
    return projects.get(current_key)


def main() -> None:
    st.set_page_config(page_title="プロット作成支援 AI", layout="wide")
    setup_logging()
    ensure_projects_loaded()
    client = init_client()

    sidebar.render_sidebar()
    view = st.session_state.get("current_view", "original")
    project = None if view in ("add_project", "manage") else get_current_project()

    if view == "add_project":
        add_project.render(client)
    elif view == "manage":
        manage.render(client)
    elif project is None:
        st.warning("プロジェクトが読み込まれていません。新しいプロジェクトを追加してください。")
    else:
        if view == "character":
            character.render(project, client)
        elif view == "plot":
            plot.render(project, client)
        elif view == "validation":
            validation.render(project, client)
        else:  # default to original view
            original.render(project, client)

    notice = st.session_state.pop("project_added_notice", None)
    if notice:
        st.success(f"新しいプロジェクト『{notice}』を追加しました。")

    if client is None and view not in ("add_project", "manage"):
        st.info("OpenAI API キーが未設定のため、生成系機能はサンプルモードで動作します。")


if __name__ == "__main__":
    main()
