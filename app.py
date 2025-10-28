# -*- coding: utf-8 -*-
import json
import hashlib
import time
import logging
import os
from pathlib import Path
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from epub_utils import extract_text_from_epub
from llm_workflow import (
    MODEL,
    CUT_MIN,
    CUT_MAX,
    TARGET_MIN,
    TARGET_MAX,
    WINDOW,
    OVERLAP,
    Panel,
    llm_cut_and_label,
    llm_plot_variants,
    build_character_brief,
    llm_character_sheet,
    text_preview,
    generate_character_graph_dot,
)

DEFAULT_STYLE_HINT = ""
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CHARACTER_FILE = DATA_DIR / "characters.json"
AUTO_INFO_FILE = DATA_DIR / "auto_characters.json"


def load_character_data() -> Dict[str, Dict[str, str]]:
    if CHARACTER_FILE.exists():
        try:
            with CHARACTER_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: {"description": str(v.get("description", ""))} for k, v in data.items()}
        except Exception:
            logging.warning("Failed to load character data from %s", CHARACTER_FILE)
    return {}


def save_character_data(characters: Dict[str, Dict[str, str]]) -> None:
    try:
        with CHARACTER_FILE.open("w", encoding="utf-8") as f:
            json.dump(characters, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.error("Failed to save character data: %s", exc)


def load_auto_character_info() -> Dict[str, Dict[str, str]]:
    if AUTO_INFO_FILE.exists():
        try:
            with AUTO_INFO_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: {"description": str(v.get("description", ""))} for k, v in data.items()}
        except Exception:
            logging.warning("Failed to load auto character info from %s", AUTO_INFO_FILE)
    return {}


def save_auto_character_info(info: Dict[str, Dict[str, str]]) -> None:
    try:
        with AUTO_INFO_FILE.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logging.error("Failed to save auto character info: %s", exc)


def trigger_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def render_table(rows):
    """pandasを使わずにMarkdownテーブルで表示"""
    if not rows:
        st.info("表示できるデータがありません。")
        return
    headers = list(rows[0].keys())

    def fmt(val):
        s = "" if val is None else str(val)
        return s.replace("|", "\\|").replace("\n", " ")

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        line = "| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |"
        lines.append(line)
    st.markdown("\n".join(lines))


def build_character_glossary_text() -> str:
    custom = st.session_state.get("custom_characters", {})
    auto = st.session_state.get("auto_characters", {})
    if not custom and not auto:
        return ""
    lines = []
    combined_names = sorted(set(list(custom.keys()) + list(auto.keys())))
    for name in combined_names:
        if name in custom:
            desc = custom[name].get("description", "").strip()
        else:
            desc = auto.get(name, {}).get("description", "").strip()
        lines.append(f"{name}: {desc or '（説明未入力）'}")
    return "\n".join(lines)


def build_dot_from_custom_characters(characters: Dict[str, Dict[str, str]]) -> str:
    if not characters:
        return ""
    lines = [
        "digraph MangaCharacters {",
        '  graph [rankdir="LR", splines=true, overlap=false];',
        '  node [shape=ellipse, style=filled, fillcolor="#f5f0ff"];'
    ]
    for name, data in sorted(characters.items()):
        desc = (data.get("description", "") or "").replace("\"", "\\\"")
        label = f"{name}" if not desc else f"{name}\\n{desc}"
        lines.append(f'  "{name}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def refresh_graph_from_custom():
    custom = st.session_state.get("custom_characters", {})
    dot = build_dot_from_custom_characters(custom)
    st.session_state.character_graph_dot = dot
    st.session_state.character_graph_nodes = sorted(custom.keys())


def format_character_sheet_markdown(md: str) -> str:
    lines = []
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            lines.append(f"**{stripped[2:].strip()}**")
        elif stripped.startswith("## "):
            lines.append(f"**{stripped[3:].strip()}**")
        else:
            lines.append(line)
    return "\n".join(lines)


def render_character_sheet(target_name: str):
    if not target_name.strip():
        st.warning("キャラクター名を指定してください。")
        return
    brief = build_character_brief(target_name.strip(), st.session_state.panels, limit=30)
    st.caption(f"該当カット数: {brief['count']}（最大30件を使用）")
    if brief["panels"]:
        render_table([{
            "id": p.id,
            "speaker": p.speaker,
            "type": p.type,
            "time": p.time,
            "tone": p.tone,
            "len": len(p.text),
            "text": text_preview(p.text, 80)
        } for p in brief["panels"]])
        with st.spinner("LLMがキャラ設定を生成中…"):
            md = llm_character_sheet(client, target_name.strip(), brief["panels"], DEFAULT_STYLE_HINT)
        st.markdown(format_character_sheet_markdown(md))
    else:
        st.warning("該当カットが見つかりませんでした。speaker・本文・entities に名前が含まれている必要があります。")




def extract_primary_characters(client, full_text: str, max_chars: int = 6000) -> Dict[str, Dict[str, str]]:
    text = (full_text or "").strip()
    if not text:
        return {}
    if len(text) > max_chars:
        text = text[:max_chars]
    prompt = """あなたは編集者アシスタントです。以下の本文から主要な登場人物を抽出し、
1) 名前
2) 50-120文字程度の説明
3) 簡潔な関係性メモ（任意）
を整理してください。最大10名まで、JSON配列のみで返してください。配列要素のスキーマは
{"name":..., "description":..., "relation":...} です。説明・関係は日本語で簡潔に。
本文:
""" + text
    try:
        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            max_output_tokens=512
        )
        raw = resp.output[0].content[0].text if getattr(resp, 'output', None) else ''
        data = _safe_json_loads(raw)
        if not data:
            alt = _safe_json_object(raw)
            if isinstance(alt, list):
                data = alt
        if not data:
            return {}
        out = {}
        for item in data:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            desc = (item.get("description", "") or "").strip()
            rel = (item.get("relation", "") or "").strip()
            combined = desc
            if rel:
                combined = f"{desc} 関係: {rel}" if desc else rel
            out[name] = {"description": combined or desc}
        return out
    except Exception as exc:
        logging.warning("Failed to extract characters: %s", exc)
        return {}


def render_character_manager():
    st.subheader("キャラクター設定管理")
    custom = st.session_state.custom_characters
    auto_info = st.session_state.auto_characters

    if "character_entry_name_input" not in st.session_state:
        st.session_state.character_entry_name_input = ""
    if "character_entry_desc_input" not in st.session_state:
        st.session_state.character_entry_desc_input = ""
    if "character_name_pick" not in st.session_state:
        st.session_state.character_name_pick = "(自由入力)"

    union_names = {
        p.speaker.strip() for p in st.session_state.get("panels", [])
        if p.speaker and p.speaker.strip().lower() not in ("", "unknown")
    }
    union_names.update(custom.keys())
    union_names.update(auto_info.keys())
    available_names = sorted(union_names)

    pick_options = ["(自由入力)"] + available_names
    current_pick = st.session_state.get("character_name_pick", "(自由入力)")
    if current_pick not in pick_options:
        current_pick = "(自由入力)"
    selected_pick = st.selectbox(
        "ラベルから選択",
        pick_options,
        index=pick_options.index(current_pick),
        key="character_name_pick"
    )
    if selected_pick != "(自由入力)":
        desc_from_sources = custom.get(selected_pick, {}).get("description", "")
        if not desc_from_sources:
            desc_from_sources = auto_info.get(selected_pick, {}).get("description", "")
        st.session_state.character_entry_name_input = selected_pick
        st.session_state.character_entry_desc_input = desc_from_sources
    name_value = st.text_input("キャラクター名", key="character_entry_name_input")
    desc_value = st.text_area("説明", key="character_entry_desc_input")

    add_cols = st.columns([1, 1])
    with add_cols[0]:
        if st.button("追加/上書き", key="save_character_entry"):
            key = name_value.strip()
            if not key:
                st.warning("キャラクター名を入力してください。")
            else:
                custom[key] = {"description": desc_value.strip()}
                save_character_data(custom)
                st.session_state.character_name_pick = key
                refresh_graph_from_custom()
                st.success(f"{key} を保存しました。")
                trigger_rerun()
    with add_cols[1]:
        if st.button("入力内容をクリア", key="clear_character_entry"):
            st.session_state.character_entry_name_input = ""
            st.session_state.character_entry_desc_input = ""
            st.session_state.character_name_pick = "(自由入力)"
            trigger_rerun()

    if custom:
        st.markdown("**登録済みキャラクター**")
        for name in sorted(custom.keys()):
            description_preview = custom[name].get("description", "") or "（説明未入力）"
            with st.expander(name, expanded=(st.session_state.character_entry_name_input == name)):
                st.write(description_preview)
                auto_desc = auto_info.get(name, {}).get("description", "")
                if auto_desc:
                    st.caption(f"自動抽出説明: {auto_desc}")
                cols = st.columns([1, 1])
                with cols[0]:
                    if st.button("編集フォームに読み込む", key=f"load_{name}"):
                        st.session_state.character_entry_name_input = name
                        st.session_state.character_entry_desc_input = custom[name].get("description", "")
                        st.session_state.character_name_pick = name if name in available_names else "(自由入力)"
                        trigger_rerun()
                with cols[1]:
                    if st.button("削除", key=f"delete_{name}"):
                        del custom[name]
                        save_character_data(custom)
                        st.session_state.auto_characters.pop(name, None)
                        save_auto_character_info(st.session_state.auto_characters)
                        refresh_graph_from_custom()
                        st.success("削除しました。")
                        trigger_rerun()
    else:
        if auto_info:
            st.markdown("**自動抽出されたキャラクター（参考）**")
            for name, data in sorted(auto_info.items()):
                st.caption(f"{name}: {data.get('description', '')}")
        else:
            st.info("登録済みのキャラクターはまだありません。")

    glossary = build_character_glossary_text()
    if st.button("ラベルの振り直し", type="primary"):
        if not st.session_state.get("source_text", "").strip():
            st.warning("本文がありません。")
        else:
            st.session_state.progress_bar = st.progress(0)
            st.session_state.progress_text = st.empty()
            custom = st.session_state.custom_characters
            panels = llm_cut_and_label(
                client,
                st.session_state.source_text,
                style_hint=DEFAULT_STYLE_HINT,
                character_glossary=glossary
            )
            st.session_state.panels = panels
            st.session_state.progress_bar.progress(1.0)
            st.session_state.progress_text.text(f"完了（{len(panels)}カット）")
            dot, nodes, desc_map = generate_character_graph_dot(client, st.session_state.source_text)
            if desc_map:
                for name, desc in desc_map.items():
                    if desc:
                        custom[name] = {"description": desc}
                save_character_data(custom)
                st.session_state.custom_characters = custom
            if dot:
                st.session_state.character_graph_dot = dot
                st.session_state.character_graph_nodes = nodes
                st.session_state.auto_characters = {name: {"description": desc_map.get(name, "")} for name in nodes}
                save_auto_character_info(st.session_state.auto_characters)
                st.success("ラベルを再生成しました。相関図も更新済みです。")
            else:
                st.session_state.auto_characters = {}
                save_auto_character_info(st.session_state.auto_characters)
                refresh_graph_from_custom()
                st.warning("ラベルを再生成しましたが、相関図を生成できませんでした。登録済みキャラクターから相関図を表示できます。")
            refresh_graph_from_custom()
            st.session_state.show_graph = False

    if st.button("戻る"):
        st.session_state.show_character_manager = False
        st.session_state.show_graph = False
        st.stop()

# --- OpenAI クライアント初期化 ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY が設定されていません。`.env` ファイルを確認してください。")
client = OpenAI(api_key=api_key)

# ===== ロギング（ターミナルにも流す） =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

loaded_custom_characters = load_character_data()
loaded_auto_info = load_auto_character_info()

if "custom_characters" not in st.session_state:
    st.session_state.custom_characters = loaded_custom_characters
if "show_character_manager" not in st.session_state:
    st.session_state.show_character_manager = False
if "character_graph_dot" not in st.session_state:
    st.session_state.character_graph_dot = ""
if "character_graph_nodes" not in st.session_state:
    st.session_state.character_graph_nodes = []
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "auto_characters" not in st.session_state:
    st.session_state.auto_characters = loaded_auto_info

if st.session_state.custom_characters and not st.session_state.character_graph_dot:
    refresh_graph_from_custom()

# ====== Streamlit UI ======
st.set_page_config(page_title="コミカライズ支援デモ", layout="wide")
st.title("📚 コミカライズ支援デモ")

# APIキー形式チェック（任意）
if not api_key.startswith("sk-") or len(api_key) < 40:
    st.error("OPENAI_API_KEY が無効です。app.py 内を修正してください。")
    st.stop()

# セッション初期化
if "panels" not in st.session_state:
    st.session_state.panels = []
if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "progress_bar" not in st.session_state:
    st.session_state.progress_bar = st.progress(0)
if "progress_text" not in st.session_state:
    st.session_state.progress_text = st.empty()

with st.sidebar:
    st.divider()
    tab = st.radio(
        "メニュー",
        ["原作解析", "プロット支援", "ネーム支援"],
        index=0,
        key="main_menu"
    )

has_panels = bool(st.session_state.get("panels"))
has_text = bool(st.session_state.get("source_text"))

def render_download_section():
    if has_text and has_panels:
        project = {
            "schema": "comicizer/v1",
            "model": MODEL,
            "cut_policy": {
                "min": CUT_MIN, "max": CUT_MAX,
                "target_min": TARGET_MIN, "target_max": TARGET_MAX
            },
            "windowing": {"window": WINDOW, "overlap": OVERLAP},
            "doc_meta": {"length": len(st.session_state["source_text"])},
            "full_text": st.session_state["source_text"],
            "panels": [p.to_dict() for p in st.session_state["panels"]],
        }
        project_json = json.dumps(project, ensure_ascii=False, indent=2)
    else:
        project_json = json.dumps({
            "schema": "comicizer/v1",
            "model": MODEL,
            "message": "保存できるデータがまだありません。（本文とパネルが必要）"
        }, ensure_ascii=False, indent=2)

    st.download_button(
        "💾 現在の結果を保存（JSON）",
        data=project_json,
        file_name="comicizer_project.json",
        mime="application/json",
        disabled=not (has_text and has_panels),
        help="本文＋パネルが生成されると有効になります。"
    )

def handle_json_upload():
    uploaded = st.file_uploader(
        "📂 前回保存したJSONを読み込む",
        type=["json"],
        key="project_uploader",
        accept_multiple_files=False
    )

    if "upload_processed" not in st.session_state:
        st.session_state.upload_processed = False

    if uploaded is not None and not st.session_state.upload_processed:
        try:
            with st.spinner("JSON を読み込み中…"):
                raw = uploaded.read().decode("utf-8")
                data = json.loads(raw)

                if data.get("schema") != "comicizer/v1":
                    st.warning("スキーマが不一致です。読み込みを中止しました。")
                else:
                    st.session_state["source_text"] = data.get("full_text", "")
                    st.session_state["panels"] = []
                    for d in data.get("panels", []):
                        st.session_state["panels"].append(Panel(
                            id=d.get("id", ""),
                            text=d.get("text", ""),
                            type=d.get("type", "unknown"),
                            speaker=d.get("speaker", "unknown"),
                            time=d.get("time", "unknown"),
                            location=d.get("location", ""),
                            scene=d.get("scene", ""),
                            tone=d.get("tone", "neutral"),
                            emotion=d.get("emotion", "neutral"),
                            action=d.get("action", ""),
                            entities=d.get("entities", []) or [],
                            source_span=d.get("source_span", {"start": -1, "end": -1}),
                            checksum=d.get("checksum", "")
                        ))
                    st.session_state.upload_processed = True
                    st.session_state.character_graph_dot = ""
                    st.session_state.character_graph_nodes = []
                    st.session_state.show_graph = False
                    existing_custom = data.get("custom_characters") or {}
                    existing_auto = data.get("auto_characters") or {}
                    st.session_state.custom_characters.update(existing_custom)
                    st.session_state.auto_characters.update(existing_auto)
                    save_character_data(st.session_state.custom_characters)
                    save_auto_character_info(st.session_state.auto_characters)
                    refresh_graph_from_custom()
                    st.success(f"読み込み完了：全文 {len(st.session_state['source_text'])} 文字 / パネル {len(st.session_state['panels'])} 件")
                    st.toast("読み込み完了", icon="✅")

        except json.JSONDecodeError as je:
            st.error(f"JSONの解析に失敗しました: {je}")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

    if st.button("🧽 アップロードをクリア"):
        st.session_state.upload_processed = False
        st.session_state["project_uploader"] = None
        st.toast("アップロードをクリアしました", icon="🧽")

if tab == "原作解析":
    st.subheader("保存/読み込み")
    render_download_section()
    handle_json_upload()

    st.write("### 1) EPUBアップロード or テキスト入力")

    epub_file = st.file_uploader(
        "📖 EPUBファイルを読み込む（本文に反映）",
        type=["epub"],
        key="epub_uploader",
        help="小説EPUBをアップロードすると本文欄にテキストを展開します。"
    )
    if "epub_processed" not in st.session_state:
        st.session_state.epub_processed = False
        st.session_state.epub_hash = ""

    if epub_file is not None:
        file_bytes = epub_file.read()
        file_hash = hashlib.sha1(file_bytes).hexdigest()
        already_loaded = st.session_state.epub_processed and st.session_state.epub_hash == file_hash
        if already_loaded:
            st.info("このEPUBは既に読み込み済みです。クリアすると再読込できます。")
        else:
            try:
                with st.spinner("EPUB をテキストに変換中…"):
                    text = extract_text_from_epub(file_bytes)
                if not text:
                    st.warning("EPUBから本文テキストを抽出できませんでした。")
                else:
                    st.session_state.source_text = text
                    st.session_state.panels = []
                    if "progress_bar" in st.session_state:
                        st.session_state.progress_bar.progress(0)
                    if "progress_text" in st.session_state:
                        st.session_state.progress_text.empty()
                    with st.spinner("主要キャラクターを抽出中…"):
                        auto_chars = extract_primary_characters(client, text)
                    if auto_chars:
                        st.session_state.auto_characters.update(auto_chars)
                    st.session_state.custom_characters.update(auto_chars)
                    save_character_data(st.session_state.custom_characters)
                    refresh_graph_from_custom()
                    st.session_state.epub_processed = True
                    st.session_state.epub_hash = file_hash
                    st.session_state.character_graph_dot = ""
                    st.session_state.character_graph_nodes = []
                    st.session_state.show_graph = False
                    refresh_graph_from_custom()
                    st.toast("EPUB取り込み完了", icon="📖")
                    st.success(f"EPUB読込完了：文字数 {len(text):,} 文字")
            except Exception as ex:
                st.error(f"EPUBの読み込みに失敗しました: {ex}")

    if st.button("🧽 EPUB読み込みをクリア", key="clear_epub"):
        st.session_state.epub_processed = False
        st.session_state.epub_hash = ""
        st.session_state["epub_uploader"] = None
        st.toast("EPUB入力をクリアしました", icon="🧽")

    src = st.text_area(
        label="本文",
        value=st.session_state.source_text or "",
        height=200,
        placeholder="ここに本文を貼り付け",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ LLMで分割＋ラベル付与を実行", type="primary"):
            if not src.strip():
                st.warning("本文を入力してください。")
            else:
                st.session_state.source_text = src
                st.session_state.progress_bar = st.progress(0)
                st.session_state.progress_text = st.empty()
                t0 = time.time()
                glossary = build_character_glossary_text()
                panels = llm_cut_and_label(
                    client,
                    src,
                    style_hint=DEFAULT_STYLE_HINT,
                    character_glossary=glossary
                )
                st.session_state.panels = panels
                st.session_state.progress_bar.progress(1.0)
                st.session_state.progress_text.text(f"完了（{len(panels)}カット）")
                logger.info(f"[USER] run_cut_and_label clicked | text_len={len(src)} | elapsed={time.time()-t0:.1f}s")
                st.session_state.character_graph_dot = ""
                st.session_state.character_graph_nodes = []
                st.session_state.show_graph = False

    with col2:
        if st.button("🧹 クリア（本文＆結果）"):
            st.session_state.source_text = ""
            st.session_state.panels = []
            st.session_state.progress_bar.progress(0)
            st.session_state.progress_text.empty()
            st.session_state.character_graph_dot = ""
            st.session_state.character_graph_nodes = []
            st.session_state.show_graph = False
            trigger_rerun()

    st.divider()
    st.write("### 2) 結果プレビュー / 絞り込み")

    if st.session_state.panels:
        fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 2])
        with fcol1:
            t_filter = st.selectbox("type", ["(all)", "dialogue", "narration", "monologue", "sfx", "stage_direction", "unknown"])
        with fcol2:
            time_filter = st.selectbox("time", ["(all)", "present", "flashback", "foreshadow", "time_skip", "unknown"])
        with fcol3:
            tone_filter = st.selectbox("tone", ["(all)", "calm", "tense", "comedic", "romantic", "tragic", "neutral"])
        with fcol4:
            q = st.text_input("キーワード検索（本文 or speaker）", "")

        def match(p: Panel) -> bool:
            if t_filter != "(all)" and p.type != t_filter:
                return False
            if time_filter != "(all)" and p.time != time_filter:
                return False
            if tone_filter != "(all)" and p.tone != tone_filter:
                return False
            if q:
                ql = q.lower()
                if (ql not in p.text.lower()) and (ql not in (p.speaker or "").lower()):
                    return False
            return True

        filtered = [p for p in st.session_state.panels if match(p)]
        st.caption(f"表示中: {len(filtered)} / 総数: {len(st.session_state.panels)}")

        rows = [{
            "id": p.id,
            "type": p.type,
            "speaker": p.speaker,
            "time": p.time,
            "location": p.location or p.scene,
            "tone": p.tone,
            "emotion": p.emotion,
            "len": len(p.text),
            "text": p.text[:120] + ("…" if len(p.text) > 120 else "")
        } for p in filtered[:30]]
        render_table(rows)
    else:
        st.info("（まだ結果はありません。本文を入力して『LLMで分割＋ラベル付与を実行』を押すか、左サイドバーから保存済みJSONを読み込んでください）")

elif tab == "プロット支援":
    st.subheader("プロット支援：区間の表現を3案で提案")
    if not st.session_state.panels:
        st.info("まずは『原作解析』タブで本文を処理してカットを生成してください。")
    else:
        total = len(st.session_state.panels)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            start_idx = st.number_input("開始インデックス（1〜N）", min_value=1, max_value=total, value=2, step=1)
        with c2:
            end_idx = st.number_input("終了インデックス（1〜N）", min_value=start_idx, max_value=total, value=min(start_idx + 4, total), step=1)
        with c3:
            run_plot = st.button("✨ 3案を生成", type="primary")

        if run_plot:
            sel = st.session_state.panels[start_idx - 1:end_idx]
            selected_text = "\n".join([p.text for p in sel])
            with st.spinner("LLMが言い換え案を生成中…"):
                variants = llm_plot_variants(client, selected_text, DEFAULT_STYLE_HINT, n=3)
            st.success(f"区間 {start_idx}〜{end_idx} に対する候補")
            for i, v in enumerate(variants, start=1):
                st.markdown(f"**案{i}**")
                st.write(v.get("variant", ""))
                if v.get("note"):
                    st.caption(f"note: {v['note']}")
                st.divider()

elif tab == "ネーム支援":
    st.subheader("ネーム支援：キャラ名で抽出 → キャラ設定を自動生成")
    if st.session_state.show_character_manager:
        render_character_manager()
    else:
        if not st.session_state.panels:
            st.info("まずは『原作解析』タブで本文を処理してカットを生成してください。")
            if st.button("キャラの設定を追加・編集", key="open_character_manager_no_panels"):
                st.session_state.show_character_manager = True
                st.session_state.show_graph = False
                st.stop()
        else:
            if st.button("キャラの設定を追加・編集", key="open_character_manager"):
                st.session_state.show_character_manager = True
                st.session_state.show_graph = False
                st.stop()

            union_names = {
                p.speaker.strip() for p in st.session_state.panels
                if p.speaker and p.speaker.strip().lower() not in ("", "unknown")
            }
            union_names.update(st.session_state.custom_characters.keys())
            union_names.update(st.session_state.auto_characters.keys())
            unique_characters = sorted(union_names)

            if "character_select_input" not in st.session_state:
                st.session_state.character_select_input = unique_characters[0] if unique_characters else ""
            if "character_select_picker" not in st.session_state:
                st.session_state.character_select_picker = "(自由入力)"

            pick_options = ["(自由入力)"] + unique_characters
            if st.session_state.character_select_picker not in pick_options:
                st.session_state.character_select_picker = "(自由入力)"
            selected_pick = st.selectbox(
                "既存キャラクターから選択",
                pick_options,
                index=pick_options.index(st.session_state.character_select_picker),
                key="character_select_picker"
            )
            if selected_pick != "(自由入力)":
                st.session_state.character_select_input = selected_pick

            char_name = st.text_input(
                "キャラクター名",
                key="character_select_input"
            )

            col_graph, col_run = st.columns([1, 1])
            with col_graph:
                label = "相関図を閉じる" if st.session_state.get("show_graph", False) else "相関図を表示"
                if st.button(label, key="toggle_graph"):
                    if st.session_state.character_graph_dot:
                        st.session_state.show_graph = not st.session_state.get("show_graph", False)
                    else:
                        st.warning("相関図を表示するには、キャラクター設定管理でラベルの振り直しを実行してください。")
                        st.session_state.show_graph = False
            with col_run:
                run_name = st.button("🧠 キャラ設定を生成")

            if st.session_state.get("show_graph", False):
                if not st.session_state.character_graph_dot and st.session_state.custom_characters:
                    refresh_graph_from_custom()
                if st.session_state.character_graph_dot:
                    st.graphviz_chart(st.session_state.character_graph_dot)
                else:
                    st.warning("相関図を表示するには、キャラクター設定管理でラベルの振り直しを実行してください。")

            if run_name and char_name.strip():
                render_character_sheet(char_name)
