# -*- coding: utf-8 -*-
import json
import re
import hashlib
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd



import os
from openai import OpenAI
from dotenv import load_dotenv  # ← これを追加
# --- .envファイルを読み込む ---
load_dotenv()
# --- 環境変数からAPIキーを取得 ---
api_key = os.getenv("OPENAI_API_KEY")
# --- キーが存在しない場合のエラーハンドリング ---
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY が設定されていません。`.env` ファイルを確認してください。")
# --- OpenAIクライアント初期化 ---
client = OpenAI(api_key=api_key)



# ===== ロギング（ターミナルにも流す） =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ===== 粒度/ウィンドウのデフォルト =====
MODEL = "gpt-5-mini"           # 変更可："gpt-4o-mini" など
CUT_MIN = 100                  # 1カット最小文字数
CUT_MAX = 220                  # 1カット最大文字数
TARGET_MIN = 150               # LLMへの理想最小
TARGET_MAX = 220               # LLMへの理想最大
WINDOW = 2000                  # LLMに渡す1チャンクの長さ（~1万文字向け）
OVERLAP = 150                  # チャンクの重なり
REQUEST_TIMEOUT = 120.0

STYLE_GUIDE = {
    "light_novel": "テンポ速め、会話中心、地の文は軽め。",
    "literary": "心理・情景を丁寧に、語彙はやや硬質。",
    "youth": "学園・青春調。自然な会話体、内面描写やや多め。",
    "suspense": "緊張感と間、伏線重視。簡潔な文で引き締める。"
}

# ===== LLM プロンプト（分割・ラベル用） =====
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

# ===== JSON配列だけ抽出 =====
JSON_ARRAY_EXTRACT = re.compile(r"\[\s*{.*}\s*\]", re.DOTALL)
def safe_json_loads(raw: str):
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
        txt = re.sub(r"\n```$", "", txt)
    m = JSON_ARRAY_EXTRACT.search(txt)
    if m:
        txt = m.group(0)
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else None
    except Exception:
        return None

# ===== LLMウィンドウ（2000/150） =====
def chunk_for_llm(text: str, window: int = WINDOW, overlap: int = OVERLAP):
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(t)
    cuts, i = [], 0
    while i < n:
        end = min(i + window, n)
        # 段落終端が後半にあれば寄せる（任意）
        brk = t[i:end].rfind("\n\n")
        if brk >= int(window * 0.5):
            end = i + brk
        cuts.append({"start": i, "end": end})
        if end >= n:
            break
        next_i = max(i + 1, end - overlap)
        i = next_i
    return cuts

# ====== データ構造 ======
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
    source_span: Dict[str, int] = None     # 原文グローバルの start/end（任意）
    checksum: str = ""
    def to_dict(self):
        d = asdict(self)
        if d["entities"] is None:
            d["entities"] = []
        if d["source_span"] is None:
            d["source_span"] = {"start": -1, "end": -1}
        return d

# ====== LLM呼び出し（Chat Completions） ======
def call_llm_chunk(chunk_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": LABEL_SYSTEM_PROMPT},
            {"role": "user", "content": f"以下の chunk を処理し、JSON配列のみを返してください。\n---\n{chunk_text}\n---"}
        ],
        timeout=REQUEST_TIMEOUT
    )
    return (resp.choices[0].message.content or "").strip()

# ====== 分割＋ラベル付け（正規化込み） ======
def llm_cut_and_label(full_text: str, style: str = "light_novel") -> List[Panel]:
    chunks = chunk_for_llm(full_text, WINDOW, OVERLAP)
    logger.info(f"[START] LLM cut&label | style={style} | text_len={len(full_text)} | chunks={len(chunks)}")
    st.session_state["progress_text"].text(f"LLM準備中… チャンク数: {len(chunks)}")
    st.session_state["progress_bar"].progress(0)

    panels: List[Panel] = []
    cut_counter = 1
    for idx, ch in enumerate(chunks, start=1):
        start_idx, end_idx = ch["start"], ch["end"]
        chunk_text = full_text[start_idx:end_idx]
        logger.info(f"[CHUNK {idx}/{len(chunks)}] span=({start_idx},{end_idx}) size={end_idx-start_idx}")

        try:
            raw = call_llm_chunk(chunk_text)
            data = safe_json_loads(raw)
            if not data:
                raise ValueError("JSON配列の抽出に失敗")

            for item in data:
                text_local = (item.get("text") or "").strip()
                if not text_local:
                    continue
                if len(text_local) > CUT_MAX:
                    text_local = text_local[:CUT_MAX]

                loc = item.get("source_local_span", {})
                ls = int(loc.get("start", 0)); le = int(loc.get("end", ls + len(text_local)))
                gs = start_idx + max(ls, 0); ge = start_idx + max(le, 0)

                pid = f"c{str(cut_counter).zfill(4)}"
                panels.append(Panel(
                    id=pid,
                    text=text_local,
                    type=item.get("type", "unknown"),
                    speaker=item.get("speaker", "unknown"),
                    time=item.get("time", "unknown"),
                    location=item.get("location", item.get("scene","")),
                    scene=item.get("scene", item.get("location","")),
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
            # 認証など致命的なものは即停止
            try:
                import openai
                if hasattr(openai, "AuthenticationError") and isinstance(ex, openai.AuthenticationError):
                    st.error("❌ OpenAI 認証に失敗しました。APIキーを確認してください。処理を中断します。")
                    logger.exception("[CHUNK %s] AuthenticationError", idx)
                    raise
            except Exception:
                pass

            # フォールバック：チャンクをそのまま1カット
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

        # 進捗UI
        st.session_state["progress_bar"].progress(idx / len(chunks))
        st.session_state["progress_text"].text(f"処理中… {idx}/{len(chunks)}")

    logger.info(f"[DONE] panels={len(panels)}")
    return panels

# ====== 追加機能：プロット支援（区間の別表現×3） ======
PLOT_SYSTEM = """あなたは編集者アシスタントです。
指定された複数カットの原文を踏まえ、意味・事実関係を変えずに、表現のみを自然で読みやすい日本語で言い換え候補を3つ作ってください。
出力は必ずJSON配列（3要素）で、各要素は {"variant": "...", "note": "..."} とします。
- variant: 提案文（1段落程度、過度に長くしない）
- note: 言い換えの狙い（視点/テンポ/語彙/情緒などの違いを短く）
文体は入力の作風に合わせるが、誇張や新規の事実追加は禁止。
"""

def llm_plot_variants(selected_text: str, style_hint: str, n: int = 3) -> List[Dict[str, str]]:
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
    data = safe_json_loads(raw)
    if not data:
        # フォールバック：そのままを返す
        return [{"variant": selected_text, "note": "フォールバック：原文そのまま"},
                {"variant": selected_text, "note": "フォールバック：原文そのまま"},
                {"variant": selected_text, "note": "フォールバック：原文そのまま"}]
    # 3つに満たなければ埋める
    out = []
    for i in range(min(len(data), n)):
        it = data[i] or {}
        out.append({"variant": it.get("variant","").strip() or selected_text,
                    "note": it.get("note","").strip() or ""})
    while len(out) < n:
        out.append({"variant": selected_text, "note": "補完"})
    return out

# ====== 追加機能：ネーム支援（キャラ名で抽出→キャラ設定生成） ======
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

def build_character_brief(name: str, panels: List[Panel], limit: int = 30) -> Dict[str, Any]:
    # name に合致するカットを抽出（speaker/本文/entities）
    name_l = name.lower()
    hits: List[Panel] = []
    for p in panels:
        if p.speaker and name_l in p.speaker.lower():
            hits.append(p); continue
        if name_l in p.text.lower():
            hits.append(p); continue
        ents = [e.lower() for e in (p.entities or [])]
        if name_l in ents:
            hits.append(p); continue
    # 重複除去（idで）
    seen = set(); uniq = []
    for p in hits:
        if p.id in seen: continue
        seen.add(p.id); uniq.append(p)
    # 多すぎる場合は時系列で先頭からlimit件
    uniq.sort(key=lambda x: (x.source_span.get("start", 0)))
    return {
        "count": len(uniq),
        "panels": uniq[:limit]
    }

def llm_character_sheet(name: str, selected_panels: List[Panel], style_hint: str) -> str:
    context = "\n\n".join([f"[{i+1}]({p.id}) speaker={p.speaker} type={p.type} time={p.time} tone={p.tone}\n{text_preview(p.text)}"
                           for i, p in enumerate(selected_panels)])
    user_prompt = f"""【キャラ名】{name}
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
    return t if len(t) <= n else t[:n] + "…"

# ====== Streamlit UI ======
st.set_page_config(page_title="コミカライズ支援デモ", layout="wide")
st.title("📚 コミカライズ支援デモ（プロット→カット自動化）")

# APIキー形式チェック（任意）
if not OPENAI_API_KEY.startswith("sk-") or len(OPENAI_API_KEY) < 40:
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
    st.header("設定")
    style = st.selectbox("作風プリセット", list(STYLE_GUIDE.keys()), index=0)
    st.caption("※ デモでは作風は軽い誘導メモ。強制ではありません。")

    st.subheader("カット粒度（最終）")
    st.text(f"MIN={CUT_MIN}, MAX={CUT_MAX}, 目安={TARGET_MIN}-{TARGET_MAX}")

    st.subheader("LLMウィンドウ（前処理）")
    st.text(f"WINDOW={WINDOW}, OVERLAP={OVERLAP}")
    st.caption("※ ~1万文字向けに最適化。5万文字なら WINDOW 1600-2200 / OVERLAP 150-200 推奨。")

st.divider()
st.subheader("保存/読み込み")

# いつでも表示。結果が無ければ disabled
has_panels = bool(st.session_state.get("panels"))
has_text   = bool(st.session_state.get("source_text"))

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
        "full_text": st.session_state["source_text"],   # 再開用に全文も保存
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

# --- ここからアップロード処理（無限ループ対策版） ---
uploaded = st.file_uploader(
    "📂 前回保存したJSONを読み込む",
    type=["json"],
    key="project_uploader",              # 明示的な key を付与
    accept_multiple_files=False
)

# アップロード処理は1回だけ通すためのフラグ
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
                # 本文とパネルを復元
                st.session_state["source_text"] = data.get("full_text", "")
                st.session_state["panels"] = []
                for d in data.get("panels", []):
                    st.session_state["panels"].append(Panel(
                        id=d.get("id",""),
                        text=d.get("text",""),
                        type=d.get("type","unknown"),
                        speaker=d.get("speaker","unknown"),
                        time=d.get("time","unknown"),
                        location=d.get("location",""),
                        scene=d.get("scene",""),
                        tone=d.get("tone","neutral"),
                        emotion=d.get("emotion","neutral"),
                        action=d.get("action",""),
                        entities=d.get("entities",[]) or [],
                        source_span=d.get("source_span",{"start":-1,"end":-1}),
                        checksum=d.get("checksum","")
                    ))
                st.session_state.upload_processed = True   # ✅ 処理済みフラグON
                st.success(f"読み込み完了：全文 {len(st.session_state['source_text'])} 文字 / パネル {len(st.session_state['panels'])} 件")
                st.toast("読み込み完了", icon="✅")

    except json.JSONDecodeError as je:
        st.error(f"JSONの解析に失敗しました: {je}")
    except Exception as e:
        st.error(f"読み込みエラー: {e}")

# アップローダを手動でクリアできるように
if st.button("🧽 アップロードをクリア"):
    st.session_state.upload_processed = False
    # ↓ 値を None に戻してウィジェットをクリア（Streamlit 1.30+ で有効）
    st.session_state["project_uploader"] = None
    st.toast("アップロードをクリアしました", icon="🧽")



st.write("### 1) 入力テキスト（~1万文字程度を想定）")
src = st.text_area(
    label="本文",
    value=st.session_state.source_text or "",
    height=200,
    placeholder="ここに本文を貼り付け",
)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("▶️ LLMで分割＋ラベル付与を実行", type="primary"):
        if not src.strip():
            st.warning("本文を入力してください。")
        else:
            st.session_state.source_text = src
            st.session_state.progress_bar = st.progress(0)
            st.session_state.progress_text = st.empty()
            t0 = time.time()
            panels = llm_cut_and_label(src, style=style)
            st.session_state.panels = panels
            st.session_state.progress_bar.progress(1.0)
            st.session_state.progress_text.text(f"完了（{len(panels)}カット）")
            logger.info(f"[USER] run_cut_and_label clicked | text_len={len(src)} | style={style} | elapsed={time.time()-t0:.1f}s")

with col2:
    if st.button("🧹 クリア（本文＆結果）"):
        st.session_state.source_text = ""
        st.session_state.panels = []
        st.session_state.progress_bar.progress(0)
        st.session_state.progress_text.empty()
        st.experimental_rerun()

st.divider()
st.write("### 2) 結果プレビュー / 絞り込み")

if st.session_state.panels:
    # フィルタ
    fcol1, fcol2, fcol3, fcol4 = st.columns([1,1,1,2])
    with fcol1:
        t_filter = st.selectbox("type", ["(all)","dialogue","narration","monologue","sfx","stage_direction","unknown"])
    with fcol2:
        time_filter = st.selectbox("time", ["(all)","present","flashback","foreshadow","time_skip","unknown"])
    with fcol3:
        tone_filter = st.selectbox("tone", ["(all)","calm","tense","comedic","romantic","tragic","neutral"])
    with fcol4:
        q = st.text_input("キーワード検索（本文 or speaker）", "")

    def match(p: Panel):
        if t_filter != "(all)" and p.type != t_filter: return False
        if time_filter != "(all)" and p.time != time_filter: return False
        if tone_filter != "(all)" and p.tone != tone_filter: return False
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
        "text": p.text[:120] + ("…" if len(p.text)>120 else "")
    } for p in filtered[:30]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.write("### 3) プロット支援：区間の表現を3案で提案")
    # 時系列（id順≒source_span.start順）でスライダー
    # パネルは生成順IDなのでそのまま並べる
    total = len(st.session_state.panels)
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        start_idx = st.number_input("開始インデックス（1〜N）", min_value=1, max_value=total, value=2, step=1)
    with c2:
        end_idx = st.number_input("終了インデックス（1〜N）", min_value=start_idx, max_value=total, value=min(start_idx+4, total), step=1)
    with c3:
        run_plot = st.button("✨ 3案を生成", type="primary")

    if run_plot:
        sel = st.session_state.panels[start_idx-1:end_idx]
        # 対象テキストを結合
        selected_text = "\n".join([p.text for p in sel])
        style_hint = STYLE_GUIDE.get(style, "")
        with st.spinner("LLMが言い換え案を生成中…"):
            variants = llm_plot_variants(selected_text, style_hint, n=3)
        st.success(f"区間 {start_idx}〜{end_idx} に対する候補")
        for i, v in enumerate(variants, start=1):
            st.markdown(f"**案{i}**")
            st.write(v.get("variant",""))
            if v.get("note"):
                st.caption(f"note: {v['note']}")
            st.divider()

    st.write("### 4) ネーム支援：キャラ名で抽出 → キャラ設定を自動生成")
    ch1, ch2 = st.columns([3,1])
    with ch1:
        char_name = st.text_input("キャラクター名", placeholder="例）佐野 / 井上 / 主人公 など")
    with ch2:
        run_name = st.button("🧠 キャラ設定を生成")
    if run_name and char_name.strip():
        brief = build_character_brief(char_name.strip(), st.session_state.panels, limit=30)
        st.caption(f"該当カット数: {brief['count']}（最大30件を使用）")
        if brief["panels"]:
            st.write(pd.DataFrame([{
                "id": p.id, "speaker": p.speaker, "type": p.type, "time": p.time,
                "tone": p.tone, "len": len(p.text), "text": text_preview(p.text, 80)
            } for p in brief["panels"]]))
            style_hint = STYLE_GUIDE.get(style, "")
            with st.spinner("LLMがキャラ設定を生成中…"):
                md = llm_character_sheet(char_name.strip(), brief["panels"], style_hint)
            st.markdown(md)
        else:
            st.warning("該当カットが見つかりませんでした。speaker・本文・entities に名前が含まれている必要があります。")

else:
    st.info("（まだ結果はありません。本文を入力して『LLMで分割＋ラベル付与を実行』を押すか、左サイドバーから保存済みJSONを読み込んでください）")
