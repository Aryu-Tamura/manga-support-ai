"""Persistence helpers for loading and saving project data."""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from . import config
from .models import EntryRecord, ProjectData


def read_json(path: Path) -> Optional[object]:
    if not path.exists():
        if path.name != config.PROJECT_INDEX_FILE.name:
            logging.warning("JSONファイルが見つかりません: %s", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        logging.error("JSONの読み込みに失敗しました: %s | %s", path, exc)
        return None


def load_character_list(path: Path) -> List[Dict[str, str]]:
    data = read_json(path)
    if isinstance(data, list):
        return [dict(item) for item in data]
    return []


def normalize_definition(definition: Dict[str, object]) -> Dict[str, object]:
    return {
        "key": str(definition.get("key")),
        "title": str(definition.get("title")),
        "panel_file": Path(definition.get("panel_file")),
        "character_file": Path(definition.get("character_file")),
    }


def load_project_definitions() -> List[Dict[str, object]]:
    definitions = [normalize_definition(d) for d in config.BASE_PROJECT_DEFINITIONS]
    extra = read_json(config.PROJECT_INDEX_FILE)
    if isinstance(extra, list):
        for item in extra:
            try:
                definitions.append(normalize_definition(item))
            except Exception as exc:
                logging.error("プロジェクト定義の読み込みに失敗: %s | %s", item, exc)
    definitions.sort(key=lambda d: d["title"])
    return definitions


def load_project(definition: Dict[str, object]) -> ProjectData:
    panel_path = Path(definition["panel_file"])
    if not panel_path.exists():
        alt = config.DATA_DIR / str(definition["key"]) / "project.json"
        if alt.exists():
            panel_path = alt

    character_path = Path(definition["character_file"])
    if not character_path.exists():
        alt_char = config.DATA_DIR / str(definition["key"]) / "characters.json"
        if alt_char.exists():
            character_path = alt_char

    raw = read_json(panel_path)
    summary = ""
    entries_data: List[Dict[str, object]] = []
    full_text = ""

    if isinstance(raw, dict):
        summary = str(raw.get("summary") or "")
        full_text = str(raw.get("full_text") or "")
        entries_data = list(raw.get("entries") or [])
    elif isinstance(raw, list):
        entries_data = raw
    elif raw is None:
        logging.warning("パネルデータが読み込めませんでした: %s", definition["panel_file"])

    entries: List[EntryRecord] = []
    for idx, item in enumerate(entries_data, start=1):
        item = dict(item or {})
        entries.append(EntryRecord.from_dict(item, fallback_id=idx))

    characters = load_character_list(character_path)
    project = ProjectData(
        key=str(definition["key"]),
        title=str(definition["title"]),
        summary=summary,
        entries=entries,
        characters=characters,
        full_text=full_text,
        source_path=panel_path,
    )
    return project


def save_project_definition(definition: Dict[str, object]) -> None:
    current = read_json(config.PROJECT_INDEX_FILE)
    if not isinstance(current, list):
        current = []
    serialized = {
        "key": definition["key"],
        "title": definition["title"],
        "panel_file": str(definition["panel_file"]),
        "character_file": str(definition["character_file"]),
    }
    current = [item for item in current if item.get("key") != serialized["key"]]
    current.append(serialized)
    config.PROJECT_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.PROJECT_INDEX_FILE.open("w", encoding="utf-8") as fp:
        json.dump(current, fp, ensure_ascii=False, indent=2)


def remove_project_definition(key: str) -> None:
    current = read_json(config.PROJECT_INDEX_FILE)
    if not isinstance(current, list):
        return
    new_items = [item for item in current if item.get("key") != key]
    if len(new_items) == len(current):
        return
    config.PROJECT_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.PROJECT_INDEX_FILE.open("w", encoding="utf-8") as fp:
        json.dump(new_items, fp, ensure_ascii=False, indent=2)


def save_project_payload(path: Path, project: ProjectData) -> None:
    payload = {
        "summary": project.summary,
        "entries": [entry.to_dict() for entry in project.entries],
        "full_text": project.full_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def save_character_file(path: Path, characters: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(characters, fp, ensure_ascii=False, indent=2)


def delete_project_files(key: str) -> None:
    project_dir = config.DATA_DIR / key
    if project_dir.exists() and project_dir.is_dir():
        shutil.rmtree(project_dir)


def ensure_projects_loaded() -> None:
    if st.session_state.get("projects"):
        return
    definitions = load_project_definitions()
    projects: Dict[str, ProjectData] = {}
    for definition in definitions:
        try:
            project = load_project(definition)
            projects[project.key] = project
        except Exception as exc:
            logging.error("プロジェクト読み込みに失敗: %s | %s", definition, exc)
    st.session_state["project_definitions"] = definitions
    st.session_state["projects"] = projects
    if definitions and "current_project" not in st.session_state:
        st.session_state["current_project"] = definitions[0]["key"]
    st.session_state.setdefault("current_view", "original")


def generate_project_key(title: str, existing_keys: List[str]) -> str:
    base = re.sub(r"[^0-9a-zA-Z]+", "_", title.strip().lower())
    base = base.strip("_") or "project"
    candidate = base
    suffix = 1
    while candidate in existing_keys:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def register_project(definition: Dict[str, object], project: ProjectData) -> None:
    projects = st.session_state.setdefault("projects", {})
    projects[project.key] = project

    definitions = st.session_state.setdefault("project_definitions", [])
    normalized = normalize_definition(definition)
    definitions = [d for d in definitions if d["key"] != project.key]
    definitions.append(normalized)
    definitions.sort(key=lambda d: d["title"])

    st.session_state["project_definitions"] = definitions
    st.session_state["current_project"] = project.key
