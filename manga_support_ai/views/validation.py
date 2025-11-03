"""Experimental UI for validating and restructuring summaries."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from ..llm_services import (
    ensure_entry_summaries,
    generate_reconstructed_summary,
    generate_summary_variations,
    should_use_llm,
)
from ..models import ProjectData, get_entry_slice


def _state_key(prefix: str, project_key: str) -> str:
    return f"validation_{prefix}_{project_key}"


def _initialise_blocks(project: ProjectData, entries) -> None:
    key = _state_key("blocks", project.key)
    current_ids = [entry.id for entry in entries]
    stored = st.session_state.get(key)
    if stored and stored.get("ids") == current_ids:
        return
    st.session_state[key] = {
        "ids": current_ids,
        "blocks": [
            {
                "entry_id": entry.id,
                "summary": entry.summary or entry.text[:120],
                "text": entry.text,
                "order": idx + 1,
            }
            for idx, entry in enumerate(entries)
        ],
    }


def _get_blocks(project: ProjectData) -> List[Dict[str, object]]:
    key = _state_key("blocks", project.key)
    return st.session_state.get(key, {}).get("blocks", [])


def render(project: ProjectData, client) -> None:
    st.header("原作理解の検証 1")
    st.caption("チャンク要約の順序や表現を編集し、新しい構成案を検証します。")

    use_client = client if should_use_llm(project, client) else None
    ensure_entry_summaries(project, use_client)

    col_start, col_end = st.columns(2)
    with col_start:
        start_idx = st.number_input("開始ID", min_value=1, max_value=max(1, project.chunk_count), value=1)
    with col_end:
        end_idx = st.number_input(
            "終了ID",
            min_value=start_idx,
            max_value=max(1, project.chunk_count),
            value=min(project.chunk_count, start_idx + 4),
        )

    entries = get_entry_slice(project, start_idx, end_idx)
    if not entries:
        st.info("該当するチャンクがありません。")
        return

    _initialise_blocks(project, entries)
    blocks = _get_blocks(project)

    st.write("### 要約ブロックの編集")
    reorder_triggered = False
    for idx, block in enumerate(blocks):
        entry_id = block["entry_id"]
        order_col, summary_col, action_col = st.columns([1, 6, 2])

        with order_col:
            order_value = st.number_input(
                "順序",
                min_value=1,
                max_value=len(blocks),
                value=int(block.get("order", idx + 1)),
                key=f"order_{project.key}_{entry_id}",
            )
            if order_value != block.get("order"):
                block["order"] = order_value
                reorder_triggered = True

        with summary_col:
            summary_value = st.text_area(
                f"要約 (ID: {entry_id})",
                value=str(block.get("summary", "")),
                key=f"summary_{project.key}_{entry_id}",
                height=100,
            )
            block["summary"] = summary_value

        with action_col:
            with st.popover("🖊️", help="表現の変更"):
                custom_prompt = st.text_area(
                    "表現変更の目的",
                    value="読みやすくする",
                    key=f"prompt_{project.key}_{entry_id}",
                    height=80,
                )
                if st.button("LLM案を生成", key=f"rewrite_{project.key}_{entry_id}"):
                    variants = generate_summary_variations(summary_value, client, custom_prompt)
                    st.session_state[_state_key("variants", f"{project.key}_{entry_id}")] = variants
                variants = st.session_state.get(_state_key("variants", f"{project.key}_{entry_id}"), [])
                if variants:
                    choice = st.selectbox(
                        "候補を選択",
                        options=variants,
                        key=f"variant_choice_{project.key}_{entry_id}",
                    )
                    if st.button("この表現を適用", key=f"apply_variant_{project.key}_{entry_id}"):
                        block["summary"] = choice
                        st.session_state[f"summary_{project.key}_{entry_id}"] = choice

            with st.popover("🗒️", help="元テキストを表示"):
                st.write(block.get("text", ""))

    if reorder_triggered or st.button("順序を整列"):
        blocks.sort(key=lambda b: b.get("order", 0))
        for i, block in enumerate(blocks, start=1):
            block["order"] = i
            st.session_state[f"order_{project.key}_{block['entry_id']}"] = i

    # 変更を ProjectData.entries に反映
    summary_map = {block["entry_id"]: block["summary"] for block in blocks}
    for entry in project.entries:
        if entry.id in summary_map:
            entry.summary = summary_map[entry.id]

    st.session_state[_state_key("blocks", project.key)]["blocks"] = blocks

    st.write("### 再構成要約の生成")
    target_length = st.select_slider(
        "目安文字数",
        options=list(range(50, 2001, 50)),
        value=300,
    )

    result_key = _state_key("result", project.key)
    if st.button("この構成で要約を生成", type="primary"):
        ordered_blocks = sorted(blocks, key=lambda b: b["order"])
        payload = [
            {"id": block["entry_id"], "summary": block["summary"]}
            for block in ordered_blocks
        ]
        result = generate_reconstructed_summary(client, payload, target_length)
        st.session_state[result_key] = result

    result = st.session_state.get(result_key)
    if result:
        st.subheader("生成された要約")
        st.write(result)
