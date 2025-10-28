# -*- coding: utf-8 -*-
"""LLMを用いたカット分割・メタ情報抽出ワークフロー"""

import json
import re
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

import streamlit as st

logger = logging.getLogger(__name__)

MODEL = "gpt-5-mini"           # 変更可："gpt-4o-mini" など
CUT_MIN = 100                  # 1カット最小文字数
CUT_MAX = 220                  # 1カット最大文字数
TARGET_MIN = 150               # LLMへの理想最小
TARGET_MAX = 220               # LLMへの理想最大
WINDOW = 2000                  # LLMに渡す1チャンクの長さ（~1万文字向け）
OVERLAP = 150                  # チャンクの重なり
REQUEST_TIMEOUT = 120.0

LABEL_SYSTEM_PROMPT = f"""あなたは編集者アシスタントです。
本文の一部（chunk）を、漫画の「1コマ」に相当するテキスト単位（カット）に分割し、
各カットへラベルを付けて JSON 配列「のみ」で返してください。

【分割ルール（厳守）】
- 狙いの長さ帯は {TARGET_MIN}〜{TARGET_MAX} 文字。
- 絶対条件: 1カットが {CUT_MAX} 文字を超える場合は必ず分割する。
- ただし SFX（擬音・ト書き）や、単独セリフ（「…」で30字以上）は {CUT_MIN} 未満でも可。
- 話者が変わる・地の文と会話が切り替わる・場面（時間/場所）が変わる・トランジション（翌朝/回想）・SFX などを境目候補とする。
- 1カット内の文数は自由（文数ではなく文字数基準で調整）。

【ラベル仕様】
- type: dialogue | narration | monologue | sfx | stage_direction | unknown
- speaker: 不明なら "unknown"
- time: present | flashback | foreshadow | time_skip | unknown
- location: 不明なら "unknown"
- tone: calm | tense | comedic | romantic | tragic | neutral
- emotion: 省略可（返さない場合は neutral 扱い）
- action: 主な動作（短い動詞句。なければ空文字）
- entities: 固有名詞（人・物・場所）の配列
- source_local_span: {{"start": 文字offset, "end": 文字offset}}  # chunk内の開始/終了（おおよそで可）
- text: カット本文（原文を必要最小限だけ整形）

【出力フォーマット（厳守）】
[
  {{
    "id_local": "k001",
    "text": "...",
    "type": "...",
    "speaker": "...",
    "time": "...",
    "location": "...",
    "tone": "...",
    "emotion": "...",
    "action": "...",
    "entities": ["...", "..."],
    "source_local_span": {{"start": 0, "end": 10}}
  }},
  ...
]
"""

PLOT_SYSTEM = """あなたは編集者アシスタントです。
指定された複数カットの原文を踏まえ、意味・事実関係を変えずに、表現のみを自然で読みやすい日本語で言い換え候補を3つ作ってください。
出力は必ずJSON配列（3要素）で、各要素は {"variant": "...", "note": "..."} とします。
- variant: 提案文（1段落程度、過度に長くしない）
- note: 言い換えの狙い（視点/テンポ/語彙/情緒などの違いを短く）
文体は入力の作風に合わせるが、誇張や新規の事実追加は禁止。
"""

NAME_SYSTEM = """あなたは編集編集者です。
与えられたカット群から、指定キャラクターの設定とネーム用の指針を整理します。
出力はMarkdownで、以下の項目を含めてください：
- # キャラクター設定（名前、役割、年齢層の推定、口調、価値観、弱み/葛藤）
- ## 登場シーン要約（時系列数点）
- ## 口調・言い回しの特徴（箇条書き）
- ## 表情・アクションの傾向（箇条書き）
- ## 主要な人間関係（文脈から推測可、過剰推測は禁止）
- ## ネーム指針（3〜5項目）
過剰な創作は避け、与えられたカット内容の範囲で推定してください。
"""

JSON_ARRAY_EXTRACT = re.compile(r"\[\s*{.*}\s*\]", re.DOTALL)
JSON_OBJECT_EXTRACT = re.compile(r"\{\s*\".*", re.DOTALL)


def _safe_json_loads(raw: str) -> Optional[List[Any]]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
        txt = re.sub(r"\n```$", "", txt)
    match = JSON_ARRAY_EXTRACT.search(txt)
    if match:
        txt = match.group(0)
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _safe_json_object(raw: str) -> Optional[Dict[str, Any]]:
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
        txt = re.sub(r"\n```$", "", txt)
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@dataclass
class Panel:
    id: str
    text: str
    type: str = "unknown"
    speaker: str = "unknown"
    time: str = "unknown"
    location: str = ""
    scene: str = ""
    tone: str = "neutral"
    emotion: str = "neutral"
    action: str = ""
    entities: List[str] = None
    source_span: Dict[str, int] = None
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data["entities"] is None:
            data["entities"] = []
        if data["source_span"] is None:
            data["source_span"] = {"start": -1, "end": -1}
        return data


def chunk_for_llm(text: str, window: int = WINDOW, overlap: int = OVERLAP) -> List[Dict[str, int]]:
    """LLMに渡すためにテキストをオーバーラップ付きで分割"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(t)
    cuts: List[Dict[str, int]] = []
    i = 0
    while i < n:
        end = min(i + window, n)
        brk = t[i:end].rfind("\n\n")
        if brk >= int(window * 0.5):
            end = i + brk
        cuts.append({"start": i, "end": end})
        if end >= n:
            break
        next_i = max(i + 1, end - overlap)
        i = next_i
    return cuts


def _call_llm_chunk(client, chunk_text: str, style_hint: str = "", extra_context: str = "") -> str:
    messages = [{"role": "system", "content": LABEL_SYSTEM_PROMPT}]
    if style_hint:
        messages.append({
            "role": "user",
            "content": f"作風ヒント（参考）:\n{style_hint}"
        })
    if extra_context:
        messages.append({
            "role": "user",
            "content": f"補足キャラクター情報（参照して一貫性を保つこと）:\n{extra_context}"
        })
    messages.append({
        "role": "user",
        "content": f"以下の chunk を処理し、JSON配列のみを返してください。\n---\n{chunk_text}\n---"
    })
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        timeout=REQUEST_TIMEOUT
    )
    return (resp.choices[0].message.content or "").strip()


def llm_cut_and_label(client, full_text: str, style_hint: str = "", character_glossary: str = "") -> List[Panel]:
    """本文をチャンクに分割しつつ LLM でカット＋ラベルを生成"""
    chunks = chunk_for_llm(full_text, WINDOW, OVERLAP)
    logger.info(f"[START] LLM cut&label | style_hint='{style_hint}' | text_len={len(full_text)} | chunks={len(chunks)}")
    st.session_state["progress_text"].text(f"LLM準備中… チャンク数: {len(chunks)}")
    st.session_state["progress_bar"].progress(0)

    panels: List[Panel] = []
    cut_counter = 1
    for idx, ch in enumerate(chunks, start=1):
        start_idx, end_idx = ch["start"], ch["end"]
        chunk_text = full_text[start_idx:end_idx]
        logger.info(f"[CHUNK {idx}/{len(chunks)}] span=({start_idx},{end_idx}) size={end_idx-start_idx}")

        try:
            raw = _call_llm_chunk(client, chunk_text, style_hint=style_hint, extra_context=character_glossary)
            data = _safe_json_loads(raw)
            if not data:
                raise ValueError("JSON配列の抽出に失敗")

            for item in data:
                text_local = (item.get("text") or "").strip()
                if not text_local:
                    continue
                if len(text_local) > CUT_MAX:
                    text_local = text_local[:CUT_MAX]

                loc = item.get("source_local_span", {})
                ls = int(loc.get("start", 0))
                le = int(loc.get("end", ls + len(text_local)))
                gs = start_idx + max(ls, 0)
                ge = start_idx + max(le, 0)

                pid = f"c{str(cut_counter).zfill(4)}"
                panels.append(Panel(
                    id=pid,
                    text=text_local,
                    type=item.get("type", "unknown"),
                    speaker=item.get("speaker", "unknown"),
                    time=item.get("time", "unknown"),
                    location=item.get("location", item.get("scene", "")),
                    scene=item.get("scene", item.get("location", "")),
                    tone=item.get("tone", "neutral"),
                    emotion=item.get("emotion", "neutral"),
                    action=item.get("action", ""),
                    entities=item.get("entities", []) or [],
                    source_span={"start": gs, "end": ge},
                    checksum="sha1:" + hashlib.sha1(text_local.encode("utf-8")).hexdigest()
                ))
                cut_counter += 1

            logger.info(f"[CHUNK {idx}] OK | cuts+={len(data)}")

        except Exception as ex:
            try:
                import openai
                if hasattr(openai, "AuthenticationError") and isinstance(ex, openai.AuthenticationError):
                    st.error("❌ OpenAI 認証に失敗しました。APIキーを確認してください。処理を中断します。")
                    logger.exception("[CHUNK %s] AuthenticationError", idx)
                    raise
            except Exception:
                pass

            pid = f"c{str(cut_counter).zfill(4)}"
            text_fallback = chunk_text.strip()[:CUT_MAX]
            panels.append(Panel(
                id=pid,
                text=text_fallback,
                type="narration",
                speaker="unknown",
                time="unknown",
                location="",
                scene="",
                tone="neutral",
                emotion="neutral",
                action="",
                entities=[],
                source_span={"start": start_idx, "end": min(start_idx + len(text_fallback), end_idx)},
                checksum="sha1:" + hashlib.sha1(text_fallback.encode("utf-8")).hexdigest()
            ))
            cut_counter += 1
            logger.warning(f"[CHUNK {idx}] JSON失敗 → フォールバック1カット: {ex}")

        st.session_state["progress_bar"].progress(idx / len(chunks))
        st.session_state["progress_text"].text(f"処理中… {idx}/{len(chunks)}")

    logger.info(f"[DONE] panels={len(panels)}")
    return panels


def llm_plot_variants(client, selected_text: str, style_hint: str, n: int = 3) -> List[Dict[str, str]]:
    """選択した区間の言い換え案を LLM で生成"""
    user_prompt = f"""【対象原文（複数カット結合）】
{selected_text}

【作風ヒント】
{style_hint}

【要件】
- 意味・事実は保持、レトリックと語順・テンポを変える
- 3案、差別化する（簡潔/情緒/テンポ重視 など）
- JSON配列のみ出力：例
[
  {{"variant":"...","note":"..."}},
  {{"variant":"...","note":"..."}},
  {{"variant":"...","note":"..."}}
]
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLOT_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        timeout=REQUEST_TIMEOUT
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = _safe_json_loads(raw)
    if not data:
        return [{"variant": selected_text, "note": "フォールバック：原文そのまま"},
                {"variant": selected_text, "note": "フォールバック：原文そのまま"},
                {"variant": selected_text, "note": "フォールバック：原文そのまま"}]
    out: List[Dict[str, str]] = []
    for i in range(min(len(data), n)):
        item = data[i] or {}
        out.append({
            "variant": (item.get("variant", "") or "").strip() or selected_text,
            "note": (item.get("note", "") or "").strip()
        })
    while len(out) < n:
        out.append({"variant": selected_text, "note": "補完"})
    return out


def build_character_brief(name: str, panels: List[Panel], limit: int = 30) -> Dict[str, Any]:
    """キャラクター名で該当カットを抽出し最大 limit 件まで返す"""
    name_l = name.lower()
    hits: List[Panel] = []
    for p in panels:
        if p.speaker and name_l in p.speaker.lower():
            hits.append(p)
            continue
        if name_l in p.text.lower():
            hits.append(p)
            continue
        ents = [e.lower() for e in (p.entities or [])]
        if name_l in ents:
            hits.append(p)
            continue
    seen = set()
    uniq: List[Panel] = []
    for p in hits:
        if p.id in seen:
            continue
        seen.add(p.id)
        uniq.append(p)
    uniq.sort(key=lambda x: (x.source_span.get("start", 0)))
    return {"count": len(uniq), "panels": uniq[:limit]}


def llm_character_sheet(client, name: str, selected_panels: List[Panel], style_hint: str) -> str:
    """キャラクター設定メモを LLM で生成"""
    context = "\n\n".join([
        f"[{i + 1}]({p.id}) speaker={p.speaker} type={p.type} time={p.time} tone={p.tone}\n{text_preview(p.text)}"
        for i, p in enumerate(selected_panels)
    ])
    user_prompt = f"""【キャラクター名】{name}
【作風ヒント】{style_hint}
【該当カット（抜粋）】
{context}

上記のみを根拠に、指定のMarkdownスキーマで出力してください。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": NAME_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        timeout=REQUEST_TIMEOUT
    )
    return (resp.choices[0].message.content or "").strip()


def text_preview(t: str, n: int = 220) -> str:
    """テキストの頭 n 文字をプレビュー用に返す"""
    return t if len(t) <= n else t[:n] + "…"


GRAPH_CHUNK_SYSTEM = """あなたは編集アシスタントです。与えられた本文の一部から登場人物と人物同士の関係を抽出し、JSONオブジェクト（1つのみ）で返してください。
JSON形式:
{
  "characters": [{"name": "...", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "label": "..."}]
}

制約:
- 同じ人物名を繰り返さない
- characters, relationships ともに最大10件程度に抑える
- 関係が無ければ relationships は空配列
- 不確かな場合は label を "" としないで簡潔な語を与える（例: 友人, 師弟, 敵対, 家族 など）
"""


def _call_graph_chunk(client, chunk_text: str) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": GRAPH_CHUNK_SYSTEM},
        {"role": "user", "content": f"本文の抜粋:\n---\n{chunk_text}\n---"}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        timeout=REQUEST_TIMEOUT
    )
    raw = (resp.choices[0].message.content or "").strip()
    return _safe_json_object(raw)


def generate_character_graph_dot(client, full_text: str) -> Tuple[str, List[str], Dict[str, str]]:
    """本文全体から人物関係を抽出し、Graphviz DOT 形式の文字列・ノード一覧・説明辞書を返す"""
    if not full_text or not full_text.strip():
        return "", [], {}

    chunks = chunk_for_llm(full_text, WINDOW, OVERLAP)
    nodes: Dict[str, str] = {}
    edges: Dict[Tuple[str, str], Set[str]] = {}

    for ch in chunks:
        chunk_text = full_text[ch["start"]:ch["end"]]
        data = _call_graph_chunk(client, chunk_text)
        if not data:
            continue

        for char in data.get("characters", []):
            name = (char.get("name") or "").strip()
            if not name:
                continue
            desc = (char.get("description") or "").strip()
            if name not in nodes or (desc and not nodes[name]):
                nodes[name] = desc

        for rel in data.get("relationships", []):
            src = (rel.get("source") or "").strip()
            tgt = (rel.get("target") or "").strip()
            label = (rel.get("label") or "").strip() or "関係"
            if not src or not tgt or src.lower() == tgt.lower():
                continue
            key = (src, tgt)
            edges.setdefault(key, set()).add(label)

    if not nodes and not edges:
        return "", [], {}

    node_names = sorted(set(list(nodes.keys()) + [s for s, _ in edges.keys()] + [t for _, t in edges.keys()]))
    lines = [
        "digraph MangaCharacters {",
        '  graph [rankdir="LR", splines=true, overlap=false];',
        '  node [shape=ellipse, style=filled, fillcolor="#f9f5ff"];'
    ]

    for name in node_names:
        lines.append(f'  "{name}" [shape=ellipse];')

    for (src, tgt), labels in edges.items():
        label_text = "\\n".join(sorted(labels))
        lines.append(f'  "{src}" -> "{tgt}" [label="{label_text}"];')

    lines.append("}")
    return "\n".join(lines), node_names, {name: nodes.get(name, "") for name in node_names}
