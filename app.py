import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
try:
    from docx import Document
except ImportError:  # pragma: no cover - optional dependency
    Document = None


DATA_DIR = Path("data")
DEFAULT_MODEL = "gpt-4o-mini"
SUMMARY_GRAIN_OPTIONS = list(range(50, 801, 50))
MAX_CONTEXT_CHARS = 4000
# LLMを使うプロジェクト（サンプルではなくAPI呼び出し）
LLM_ENABLED_PROJECT_KEYS = {"project1"}
PROJECT_DEFINITIONS = [
    {
        "key": "project1",
        "title": "銀河鉄道の夜",
        "panel_file": DATA_DIR / "gingatetudono_yoru_labeled.json",
        "character_file": DATA_DIR / "character_gingatetudonoyoru.json",
    },
    {
        "key": "project2",
        "title": "井上尚弥の書籍",
        "panel_file": DATA_DIR / "inouenaoya_labeled.json",
        "character_file": DATA_DIR / "character_inouenaoya.json",
    },
]


@dataclass
class PanelRecord:
    id: str
    text: str
    type: str
    speakers: List[str]
    time: str
    location: str
    action: str
    source_span: Dict[str, int]
    checksum: str

    @property
    def index(self) -> int:
        """c0001 -> 1"""
        digits = "".join(ch for ch in self.id if ch.isdigit())
        return int(digits) if digits else 0


@dataclass
class ProjectData:
    key: str
    title: str
    panels: List[PanelRecord]
    characters: List[Dict[str, str]]

    @property
    def chunk_count(self) -> int:
        return len(self.panels)

    @property
    def full_text(self) -> str:
        return "\n\n".join(panel.text for panel in self.panels)


def should_use_llm(project: ProjectData, client: Optional[OpenAI]) -> bool:
    """プロジェクトとAPIクライアントの有無からLLMを利用するか判定"""
    return client is not None and project.key in LLM_ENABLED_PROJECT_KEYS


def emphasize_character_names(text: str, project: ProjectData) -> str:
    """要約文中のキャラクター名を太字にする"""
    if not text:
        return ""
    names = [c.get("Name") for c in project.characters if c.get("Name")]
    for name in sorted(set(names), key=len, reverse=True):
        if not name:
            continue
        pattern = re.compile(rf"(?<!\*){re.escape(name)}(?!\*)")
        text = pattern.sub(lambda m: f"**{m.group(0)}**", text)
    return text


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        logging.warning("JSONファイルが見つかりません: %s", path)
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        logging.error("JSONの読み込みに失敗しました: %s | %s", path, exc)
    return []


def parse_panel(item: Dict[str, Any]) -> PanelRecord:
    return PanelRecord(
        id=str(item.get("id") or ""),
        text=str(item.get("text") or ""),
        type=str(item.get("type") or "unknown"),
        speakers=[str(s) for s in (item.get("speakers") or [])],
        time=str(item.get("time") or "unknown"),
        location=str(item.get("location") or ""),
        action=str(item.get("action") or ""),
        source_span=item.get("source_span") or {},
        checksum=str(item.get("checksum") or ""),
    )


def load_project(definition: Dict[str, Any]) -> ProjectData:
    panels_raw = load_json_list(definition["panel_file"])
    panels = sorted(
        (parse_panel(item) for item in panels_raw),
        key=lambda p: p.index,
    )
    characters = load_json_list(definition["character_file"])
    return ProjectData(
        key=definition["key"],
        title=definition["title"],
        panels=panels,
        characters=characters,
    )


def ensure_projects_loaded() -> None:
    if "projects" in st.session_state:
        return
    projects = {}
    for definition in PROJECT_DEFINITIONS:
        projects[definition["key"]] = load_project(definition)
    st.session_state["projects"] = projects
    st.session_state["current_project"] = PROJECT_DEFINITIONS[0]["key"]
    st.session_state["current_view"] = "original"


def init_client() -> Optional[OpenAI]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY が設定されていません。`.env` を確認してください。")
        return None
    if len(api_key) < 32:
        st.warning("APIキーの形式が短すぎるようです。必要な権限があるか確認してください。")
    try:
        client = OpenAI(api_key=api_key)
    except Exception as exc:
        st.error(f"OpenAI クライアントの初期化に失敗しました: {exc}")
        return None
    return client


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_current_project() -> ProjectData:
    key = st.session_state.get("current_project", PROJECT_DEFINITIONS[0]["key"])
    return st.session_state["projects"][key]


def render_sidebar() -> None:
    st.sidebar.markdown("### 📖 プロット作成支援 AI")

    project_titles = {p["key"]: p["title"] for p in PROJECT_DEFINITIONS}
    project_keys = list(project_titles.keys())
    current_project_key = st.session_state.get("current_project", project_keys[0])
    current_index = project_keys.index(current_project_key)
    selected_key = st.sidebar.selectbox(
        "プロジェクトを選択",
        options=project_keys,
        index=current_index,
        format_func=lambda k: project_titles[k],
    )
    if selected_key != current_project_key:
        st.session_state["current_project"] = selected_key

    st.sidebar.divider()
    if st.sidebar.button("原作理解", use_container_width=True):
        st.session_state["current_view"] = "original"
    if st.sidebar.button("キャラ解析", use_container_width=True):
        st.session_state["current_view"] = "character"
    if st.sidebar.button("プロット支援", use_container_width=True):
        st.session_state["current_view"] = "plot"

    st.sidebar.divider()
    st.sidebar.markdown("#### 👤 Demo User")
    st.sidebar.button("プロジェクトを追加する", use_container_width=True, disabled=True)
    st.sidebar.caption("デモのため、新規プロジェクト作成は現在できません。")


def get_panel_slice(project: ProjectData, start: int, end: int) -> List[PanelRecord]:
    if project.chunk_count == 0:
        return []
    start_idx = max(1, start)
    end_idx = min(project.chunk_count, end)
    if start_idx > end_idx:
        return []
    return project.panels[start_idx - 1:end_idx]


def panels_to_context(panels: List[PanelRecord], limit_chars: int = MAX_CONTEXT_CHARS) -> str:
    joined = "\n\n".join(f"[{panel.id}] {panel.text}" for panel in panels)
    return joined[:limit_chars]


def extract_text_response(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "output_text"):
        return response.output_text.strip()
    output = getattr(response, "output", None)
    if isinstance(output, list) and output:
        content = output[0].get("content")
        if isinstance(content, list) and content and "text" in content[0]:
            return str(content[0]["text"]).strip()
    return ""


def call_responses_api(client: OpenAI, system_prompt: str, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}],
        })
    messages.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_prompt}],
    })
    return client.responses.create(
        model=DEFAULT_MODEL,
        input=messages,
    )


def summarize_section(client: Optional[OpenAI], project: ProjectData, panels: List[PanelRecord], granularity: int) -> str:
    if not panels:
        return "対象のチャンクが選択されていません。"
    preview_span = f"{panels[0].id}〜{panels[-1].id}"
    joined_context = panels_to_context(panels)
    if not should_use_llm(project, client):
        return (
            f"【サンプル要約】\n"
            f"{project.title} の {preview_span} を約{granularity}文字で要約した例を表示しています。\n"
            "このプロジェクトではサンプル要約を使用しています。"
        )
    system_prompt = (
        "あなたは小説編集アシスタントです。指定されたテキスト断片を読んで、"
        "重要な出来事・登場人物・感情の流れを押さえながら、指定された文字数目安で日本語要約を作成してください。"
    )
    user_prompt = (
        f"作品: {project.title}\n"
        f"対象チャンク: {preview_span}（全{project.chunk_count}チャンク）\n"
        f"目安文字数: 約{granularity}文字\n"
        "テキスト:\n"
        f"{joined_context}\n"
        "----\n"
        "要約のみを日本語で出力してください。"
    )
    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        summary = extract_text_response(response)
        return summary or f"要約の生成に失敗しました（{preview_span}）。"
    except Exception as exc:
        logging.error("要約生成に失敗しました: %s", exc)
        return f"要約生成でエラーが発生しました: {exc}"


def find_character_contexts(project: ProjectData, name: str, limit: int = 3) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    target = name.strip()
    if not target:
        return hits
    for panel in project.panels:
        if target in panel.text:
            snippet = panel.text.strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            hits.append((panel.id, snippet))
        if len(hits) >= limit:
            break
    return hits


def generate_character_analysis(
    client: Optional[OpenAI],
    project: ProjectData,
    character: Dict[str, str],
) -> str:
    name = character.get("Name", "（名称不明）")
    role = character.get("Role", "")
    details = character.get("Details", "")
    contexts = find_character_contexts(project, name)
    if not should_use_llm(project, client):
        lines = [
            f"【サンプル設定メモ】{name}",
            f"- 役割: {role or '未設定'}",
            f"- 詳細: {details[:200]}{'…' if len(details) > 200 else ''}",
            "- 参考チャンク: " + ", ".join(pid for pid, _ in contexts) if contexts else "- 参考チャンク: なし",
        ]
        return "\n".join(lines)

    context_text = "\n".join(f"[{pid}] {snippet}" for pid, snippet in contexts) or "（本文参照なし）"
    system_prompt = (
        "あなたは漫画制作のキャラクター監修アシスタントです。"
        "提供された役割説明と本文抜粋をもとに、編集者向けのキャラクターメモを簡潔にまとめてください。"
    )
    user_prompt = (
        f"作品: {project.title}\n"
        f"キャラクター名: {name}\n"
        f"役割: {role}\n"
        f"人物詳細メモ:\n{details}\n"
        "参考本文抜粋:\n"
        f"{context_text}\n"
        "----\n"
        "以下の構成で日本語出力してください:\n"
        "1. キャラクター概要（2〜3文）\n"
        "2. 性格・価値観\n"
        "3. 技能/強みと弱み\n"
        "4. 関係性メモ（本文から推測できる範囲）"
    )
    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        result = extract_text_response(response)
        return result or f"{name} の解析結果を生成できませんでした。"
    except Exception as exc:
        logging.error("キャラ解析に失敗しました: %s", exc)
        return f"キャラクター解析でエラーが発生しました: {exc}"


def generate_plot_script(
    client: Optional[OpenAI],
    project: ProjectData,
    panels: List[PanelRecord],
    characters: List[Dict[str, str]],
) -> str:
    if not panels:
        return "チャンクが選択されていません。"
    range_label = f"{panels[0].id}〜{panels[-1].id}"
    character_names = [c.get("Name", "") for c in characters if c.get("Name")]
    speakers = ", ".join(character_names[:10]) or "（サンプル）"
    context = panels_to_context(panels)
    if not should_use_llm(project, client):
        sample_dialogue = [
            f"【サンプルプロット】範囲: {range_label}",
            "ナレーション：「ここに場面説明が入ります」",
            "登場人物A：「セリフ例：状況を伝えるセリフ」",
            "登場人物B：「セリフ例：リアクションのセリフ」",
            "ナレーション：「次の場面転換を示す描写」",
        ]
        return "\n".join(sample_dialogue)

    system_prompt = (
        "あなたは漫画ネーム制作の脚本アシスタントです。"
        "提供された本文チャンクを参考に、会話主体のシナリオ形式で叩き台を作成してください。"
    )
    user_prompt = (
        f"作品: {project.title}\n"
        f"対象チャンク: {range_label}\n"
        f"利用可能なキャラクター候補: {speakers}\n"
        "本文抜粋:\n"
        f"{context}\n"
        "----\n"
        "要件:\n"
        "- 話者名：「セリフ」の形式で記述\n"
        "- 場面転換やアクションは1行で簡潔に\n"
        "- 未登場キャラでも本文から推測できる範囲で登場可\n"
        "- 全体で10行前後を目安\n"
        "- 最後に次の検討ポイントを箇条書きで2項目程度"
    )
    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        script = extract_text_response(response)
        return script or f"{range_label} のプロットを生成できませんでした。"
    except Exception as exc:
        logging.error("プロット生成に失敗しました: %s", exc)
        return f"プロット生成でエラーが発生しました: {exc}"


def create_docx_bytes(script: str) -> Optional[bytes]:
    if not script.strip():
        return None
    if Document is None:
        return None
    document = Document()
    for line in script.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.read()


def render_original_overview(project: ProjectData, client: Optional[OpenAI]) -> None:
    st.header("原作理解")
    llm_mode = should_use_llm(project, client)
    if llm_mode:
        st.caption("OpenAI API を用いて指定区間の要約を生成します。")
    else:
        st.caption("このプロジェクトではサンプル要約を表示します。区間と粒度を指定して体験してください。")
    st.write(f"チャンク総数: {project.chunk_count}")

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
        default_grain = 200 if 200 in SUMMARY_GRAIN_OPTIONS else SUMMARY_GRAIN_OPTIONS[0]
        grain = st.select_slider("粒度（目安文字数）", options=SUMMARY_GRAIN_OPTIONS, value=default_grain)

    panels = get_panel_slice(project, start_idx, end_idx)
    st.markdown(f"選択中のチャンク: {panels[0].id if panels else 'なし'} 〜 {panels[-1].id if panels else 'なし'}")

    summary_state_key = f"latest_summary_{project.key}"
    if st.button("要約を生成", type="primary"):
        summary = summarize_section(client, project, panels, grain)
        st.session_state[summary_state_key] = summary

    summary_value = st.session_state.get(summary_state_key)
    if summary_value:
        st.subheader("要約結果")
        formatted_summary = emphasize_character_names(summary_value, project)
        st.markdown(formatted_summary.replace("\n", "  \n"))

    with st.expander("参考：選択チャンクの本文（上限あり）", expanded=False):
        if panels:
            st.write(panels_to_context(panels))
        else:
            st.info("チャンクが選択されていません。")


def render_character_view(project: ProjectData, client: Optional[OpenAI]) -> None:
    st.header("キャラ解析")
    llm_mode = should_use_llm(project, client)
    if llm_mode:
        st.caption("OpenAI API を使って選択キャラクターの解析メモを生成します。")
    else:
        st.caption("このプロジェクトではサンプルのキャラ解析結果を表示します。")

    if not project.characters:
        st.warning("キャラクターデータが登録されていません。")
        return

    name_to_character = {c.get("Name", f"キャラ{i}"): c for i, c in enumerate(project.characters, start=1)}
    character_names = list(name_to_character.keys())
    default_index = 0
    selected_name = st.selectbox(
        "キャラクターを選択",
        options=character_names,
        index=default_index,
    )
    character = name_to_character[selected_name]

    st.subheader("キャラクター情報")
    st.markdown(f"**名前**: {character.get('Name', '不明')}")
    st.markdown(f"**役割**: {character.get('Role', '不明')}")
    st.markdown("**詳細メモ**")
    st.write(character.get("Details", "（説明なし）"))

    analysis_state_key = f"latest_character_analysis_{project.key}_{selected_name}"
    if st.button("キャラ解析を生成", type="primary"):
        result = generate_character_analysis(client, project, character)
        st.session_state[analysis_state_key] = result

    analysis_value = st.session_state.get(analysis_state_key)
    if analysis_value:
        st.subheader("解析結果")
        st.write(analysis_value)

    contexts = find_character_contexts(project, character.get("Name", ""))
    with st.expander("参考：本文抜粋（最大3件）", expanded=False):
        if contexts:
            for pid, snippet in contexts:
                st.markdown(f"- **{pid}**: {snippet}")
        else:
            st.info("本文内に該当キャラの記述が見つかりませんでした。")


def render_plot_view(project: ProjectData, client: Optional[OpenAI]) -> None:
    st.header("プロット支援")
    llm_mode = should_use_llm(project, client)
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
        end_idx = st.number_input("終了チャンク", min_value=start_idx, max_value=project.chunk_count, value=min(project.chunk_count, start_idx + 4))

    panels = get_panel_slice(project, start_idx, end_idx)
    st.markdown(f"選択チャンク: {panels[0].id if panels else 'なし'} 〜 {panels[-1].id if panels else 'なし'}")

    plot_state_key = f"latest_plot_script_{project.key}"
    edit_key = f"plot_editor_{project.key}"
    if st.button("プロット叩き台を生成", type="primary"):
        script = generate_plot_script(client, project, panels, project.characters)
        st.session_state[plot_state_key] = script
        st.session_state[edit_key] = script

    script_text = st.session_state.get(plot_state_key)
    if script_text is not None:
        st.subheader("生成結果")
        if edit_key not in st.session_state:
            st.session_state[edit_key] = script_text

        col_script, col_source = st.columns([3, 2])
        with col_script:
            st.text_area(
                "シナリオ案（編集可）",
                key=edit_key,
                height=420,
            )
            edited_script = st.session_state.get(edit_key, "")
            if Document is None:
                st.info("Word形式でのダウンロードには `python-docx` のインストールが必要です。")
            else:
                docx_bytes = create_docx_bytes(edited_script)
                if docx_bytes:
                    st.download_button(
                        "Wordファイルとしてダウンロード",
                        data=docx_bytes,
                        file_name=f"{project.key}_plot.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
        with col_source:
            st.text_area(
                "本文（リファレンス）",
                value=project.full_text,
                height=420,
                disabled=True,
            )


def main() -> None:
    st.set_page_config(page_title="プロット作成支援 AI", layout="wide")
    setup_logging()
    ensure_projects_loaded()
    client = init_client()
    render_sidebar()
    project = get_current_project()

    view = st.session_state.get("current_view", "original")
    if view == "original":
        render_original_overview(project, client)
    elif view == "character":
        render_character_view(project, client)
    elif view == "plot":
        render_plot_view(project, client)
    else:
        st.warning("ビューを特定できませんでした。原作理解を表示します。")
        st.session_state["current_view"] = "original"
        render_original_overview(project, client)

    if client is None:
        st.stop()


if __name__ == "__main__":
    main()
