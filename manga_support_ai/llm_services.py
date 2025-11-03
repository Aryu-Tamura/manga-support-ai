"""High level interactions with OpenAI APIs."""

import os
import logging
import re
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

from . import config
from .exceptions import LLMUnavailableError
from .models import EntryRecord, ProjectData, entries_to_context, find_character_contexts
from .utils import read_json_from_string


def _get_secret(key: str) -> Optional[str]:
    try:
        return st.secrets[key]
    except Exception:
        return None


def init_client() -> Optional[OpenAI]:
    load_dotenv()
    api_key = _get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY が設定されていません。`.env` または Streamlit Secrets を確認してください。")
        return None
    if len(api_key) < 32:
        st.warning("APIキーの形式が短すぎる可能性があります。権限を確認してください。")
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover - network failure
        st.error(f"OpenAI クライアントの初期化に失敗しました: {exc}")
        return None


def should_use_llm(project: ProjectData, client: Optional[OpenAI]) -> bool:
    return client is not None and project.key not in config.SAMPLE_ONLY_PROJECT_KEYS


def emphasize_character_names(text: str, project: ProjectData) -> str:
    if not text:
        return ""
    names = [c.get("Name") for c in project.characters if c.get("Name")]
    for name in sorted(set(names), key=len, reverse=True):
        if not name:
            continue
        pattern = re.compile(rf"(?<!\*){re.escape(name)}(?!\*)")
        text = pattern.sub(lambda m: f"**{m.group(0)}**", text)
    return text


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
    try:
        return client.responses.create(model=config.DEFAULT_MODEL, input=messages)
    except RateLimitError as exc:  # pragma: no cover - network failure
        raise LLMUnavailableError(str(exc)) from exc
    except APIError as exc:  # pragma: no cover - network failure
        raise LLMUnavailableError(str(exc)) from exc


def extract_text_response(response) -> str:
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


def summarize_section(
    client: Optional[OpenAI],
    project: ProjectData,
    entries: List[EntryRecord],
    granularity: int,
) -> str:
    if not entries:
        return "対象のチャンクが選択されていません。"
    preview_span = f"{entries[0].id}〜{entries[-1].id}"
    joined_context = entries_to_context(entries, config.MAX_CONTEXT_CHARS)
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
        response = call_responses_api(client, system_prompt, user_prompt)  # type: ignore[arg-type]
        summary = extract_text_response(response)
        return summary or f"要約の生成に失敗しました（{preview_span}）。"
    except Exception as exc:  # pragma: no cover - network failure
        logging.error("要約生成に失敗しました: %s", exc)
        return f"要約生成でエラーが発生しました: {exc}"


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
            "- 参考チャンク: " + ", ".join(str(pid) for pid, _ in contexts)
            if contexts
            else "- 参考チャンク: なし",
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
        response = call_responses_api(client, system_prompt, user_prompt)  # type: ignore[arg-type]
        result = extract_text_response(response)
        return result or f"{name} の解析結果を生成できませんでした。"
    except Exception as exc:  # pragma: no cover - network failure
        logging.error("キャラ解析に失敗しました: %s", exc)
        return f"キャラクター解析でエラーが発生しました: {exc}"


def generate_plot_script(
    client: Optional[OpenAI],
    project: ProjectData,
    entries: List[EntryRecord],
    characters: List[Dict[str, str]],
) -> str:
    if not entries:
        return "チャンクが選択されていません。"
    range_label = f"{entries[0].id}〜{entries[-1].id}"
    character_names = [c.get("Name", "") for c in characters if c.get("Name")]
    speakers = ", ".join(character_names[:10]) or "（サンプル）"
    context = entries_to_context(entries, config.MAX_CONTEXT_CHARS)

    if not should_use_llm(project, client):
        sample_dialogue = [
            f"【サンプルプロット】範囲: {range_label}",
            "Scene 1（導入）",
            "ナレーション：「舞台設定と感情トーンを描写し、これから出てくる発話の流れを補助します。」",
            "キャラクターA：「原文にある発話内容を、漫画向けに言い回しだけ整えて書きます。」",
            "キャラクターB：「相手の発言に対する反応も含め、原文の台詞を削らずに掲載します。」",
            "ナレーション：「小さなアクションや仕草を補足し、文の冗長さだけ軽く整えます。」",
            "",
            "Scene 2（展開）",
            "キャラクターC：「原文の台詞をベースに、言い回しだけ調整して書きます。」",
            "キャラクターA：「会話の流れが分かるよう、原文で話している順番を保ってください。」",
            "ナレーション：「会話の合間を補足し、次の展開に繋がる仕草や視線を短く添えます。」",
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
        "- 場面が変わる際は **場面タイトル** を書き、続けて2〜3行で環境・感情・小道具を描写する\n"
        "- 原文に登場する発話者・セリフは漏らさず全て登場させ、順序も極力維持する（必要に応じて言い回しのみ整える）\n"
        "- 会話の間には必要最低限のト書きや状況描写を入れて漫画用に補助するが、要約にはしない\n"
        "- 選択範囲に含まれる出来事・キャラクター・伏線を時系列で網羅し、重要なアクションはト書きで補足する\n"
        "- 最後に次の検討ポイントを3項目、箇条書きで提示する"
    )

    try:
        response = call_responses_api(client, system_prompt, user_prompt)  # type: ignore[arg-type]
        script = extract_text_response(response)
        return script or f"{range_label} のプロットを生成できませんでした。"
    except Exception as exc:  # pragma: no cover - network failure
        logging.error("プロット生成に失敗しました: %s", exc)
        return f"プロット生成でエラーが発生しました: {exc}"


def _local_overall_summary(full_text: str, limit: int = 600) -> str:
    text = full_text.strip()
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    preview = " ".join(paragraphs[:3])
    return preview[:limit] + ("…" if len(preview) > limit else "")


def generate_overall_summary(client: OpenAI, title: str, full_text: str) -> str:
    excerpt = full_text.strip()
    if len(excerpt) > 12000:
        excerpt = excerpt[:12000]
    system_prompt = (
        "あなたは小説編集者です。与えられた作品本文を踏まえ、"
        "主要な出来事・舞台・主要登場人物・対立構造を含む400〜600文字の日本語要約を作成してください。"
    )
    user_prompt = (
        f"作品タイトル: {title}\n"
        "本文抜粋（最大12000字）:\n"
        f"{excerpt}\n"
        "----\n"
        "上記を踏まえた作品全体の要約を1段落で出力してください。"
    )
    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        summary = extract_text_response(response).strip()
        if summary:
            return summary
    except LLMUnavailableError as exc:  # pragma: no cover - network failure
        logging.warning("作品要約の生成に失敗: %s", exc)
        raise
    except Exception as exc:  # pragma: no cover - other API errors
        logging.warning("作品要約の生成でエラー: %s", exc)
    return _local_overall_summary(excerpt)


def extract_primary_characters(
    client: OpenAI,
    full_text: str,
    max_chars: int = 6000,
) -> List[Dict[str, str]]:
    excerpt = full_text.strip()
    if not excerpt:
        return []
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    system_prompt = (
        "あなたは編集者アシスタントです。本文から主要な登場人物を抽出し、"
        "名前・役割・詳細メモを整理してください。"
    )
    user_prompt = (
        "以下の本文から最多10名の登場人物を抽出し、JSON配列で返してください。\n"
        "各要素は {\"Name\": \"名前\", \"Role\": \"役割\", \"Details\": \"説明\"} とします。\n"
        "説明は日本語で120文字以内にしてください。\n"
        "本文:\n"
        f"{excerpt}"
    )

    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        raw = extract_text_response(response)
        data = read_json_from_string(raw)
        if isinstance(data, list):
            characters: List[Dict[str, str]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("Name") or item.get("name") or "").strip()
                if not name:
                    continue
                characters.append(
                    {
                        "Name": name,
                        "Role": str(item.get("Role") or item.get("role") or ""),
                        "Details": str(item.get("Details") or item.get("details") or ""),
                    }
                )
            return characters
    except Exception as exc:  # pragma: no cover - network failure
        logging.warning("キャラクター抽出に失敗: %s", exc)
    return []


def ensure_entry_summaries(
    project: ProjectData,
    client: Optional[OpenAI],
    target_length: int = 120,
) -> bool:
    """Fill missing summaries for entries. Returns True if any summary was added."""
    updated = False
    for entry in project.entries:
        if entry.summary.strip():
            continue
        entry.summary = generate_entry_summary(entry.text, client, target_length)
        updated = True
    return updated


def generate_entry_summary(
    text: str,
    client: Optional[OpenAI],
    target_length: int = 120,
) -> str:
    fallback = text.strip()
    if not fallback:
        return ""
    fallback = fallback[:target_length] + ("…" if len(fallback) > target_length else "")
    if client is None:
        return fallback

    system_prompt = (
        "あなたは編集者アシスタントです。入力された本文を1〜2文、"
        "約{}文字で要約してください。重要な固有名詞と出来事を残し、過剰な創作はしないこと。"
    ).format(target_length)
    user_prompt = f"本文:\n{text}\n----\n要約のみを日本語で出力してください。"

    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        summary = extract_text_response(response).strip()
        return summary or fallback
    except Exception as exc:  # pragma: no cover - network failure
        logging.warning("エントリー要約生成に失敗: %s", exc)
        return fallback


def generate_summary_variations(
    summary: str,
    client: Optional[OpenAI],
    prompt: str,
    count: int = 3,
) -> List[str]:
    if not client:
        return [summary]
    system_prompt = (
        "あなたは編集者アシスタントです。与えられた要約文を目的に沿って言い換えてください。"
        "意味を変えず、文章量は同程度に保ってください。"
    )
    user_prompt = (
        f"元の要約:\n{summary}\n"
        f"編集目的:\n{prompt or '読みやすくする'}\n"
        "----\n"
        f"{count}通りの言い換えをJSON配列で返してください。各要素は{{\"variant\": \"...\"}}とします。"
    )
    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        raw = extract_text_response(response)
        data = read_json_from_string(raw)
        if isinstance(data, list):
            variants = []
            for item in data[:count]:
                if isinstance(item, dict) and item.get("variant"):
                    variants.append(str(item["variant"]).strip())
            return variants or [summary]
    except Exception as exc:  # pragma: no cover
        logging.warning("言い換え生成に失敗: %s", exc)
    return [summary]


def generate_reconstructed_summary(
    client: Optional[OpenAI],
    blocks: List[Dict[str, object]],
    target_length: int,
) -> str:
    combined = "\n".join(
        f"[{block.get('id')}] {block.get('summary', '')}" for block in blocks
    )
    if not client:
        text = " ".join(str(block.get("summary", "")) for block in blocks)
        return text[:target_length] + ("…" if len(text) > target_length else "")

    system_prompt = (
        "あなたは編集者です。提示された要約群を指定順序のまままとめ直し、"
        "約{}文字で新しい要約文を作成してください。重要な情報を維持し、順序を尊重すること。"
    ).format(target_length)
    user_prompt = (
        f"要約ブロック（順序通り）:\n{combined}\n"
        "----\n"
        "一つの要約文として再構成し、日本語で出力してください。"
    )
    try:
        response = call_responses_api(client, system_prompt, user_prompt)
        result = extract_text_response(response).strip()
        return result or "（要約の再構成に失敗しました）"
    except Exception as exc:  # pragma: no cover
        logging.error("再構成要約の生成に失敗: %s", exc)
        return "（要約の再構成に失敗しました）"


def build_character_glossary(characters: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for character in characters:
        name = (character.get("Name") or "").strip()
        if not name:
            continue
        details = (character.get("Details") or character.get("Role") or "").strip()
        entry = f"{name}: {details}" if details else name
        lines.append(entry)
    return "\n".join(lines)
