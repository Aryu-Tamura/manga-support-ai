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
    orders = [block.get("order", idx + 1) for idx, block in enumerate(blocks)]
    block_rows = [
        {
            "order": order,
            "entry_id": block["entry_id"],
            "summary": block.get("summary", ""),
        }
        for order, block in zip(orders, blocks)
    ]
    edited = st.data_editor(
        block_rows,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "order": st.column_config.NumberColumn("順序", min_value=1, max_value=len(blocks), step=1),
            "entry_id": st.column_config.NumberColumn("ID", disabled=True),
            "summary": st.column_config.TextColumn("要約", max_chars=600),
        },
        key=f"validation_editor_{project.key}",
        use_container_width=True,
    )

    edited.sort(key=lambda row: row["order"])
    id_to_block = {block["entry_id"]: block for block in blocks}
    new_blocks: List[Dict[str, object]] = []
    for idx, row in enumerate(edited, start=1):
        block = id_to_block[row["entry_id"]]
        block["order"] = idx
        block["summary"] = row["summary"]
        new_blocks.append(block)
    blocks[:] = new_blocks

    selected_id = st.selectbox(
        "詳細を編集する要約ブロック",
        options=[block["entry_id"] for block in blocks],
        format_func=lambda bid: f"ID {bid}",
        key=f"validation_select_{project.key}",
    )
    active_block = next(block for block in blocks if block["entry_id"] == selected_id)

    st.markdown("**要約の編集**")
    edited_summary = st.text_area(
        "要約",
        value=active_block.get("summary", ""),
        key=f"validation_summary_edit_{project.key}_{selected_id}",
        height=160,
    )
    active_block["summary"] = edited_summary

    with st.expander("表現の変更 (LLM)"):
        custom_prompt = st.text_area(
            "表現変更の目的",
            value="読みやすくする",
            key=f"validation_prompt_{project.key}_{selected_id}",
            height=100,
        )
        if st.button("LLM案を生成", key=f"validation_generate_{project.key}_{selected_id}"):
            variants = generate_summary_variations(edited_summary, client, custom_prompt)
            st.session_state[_state_key("variants", f"{project.key}_{selected_id}")] = variants
        variants = st.session_state.get(_state_key("variants", f"{project.key}_{selected_id}"), [])
        if variants:
            choice = st.selectbox(
                "候補を選択",
                options=variants,
                key=f"validation_variant_choice_{project.key}_{selected_id}",
            )
            if st.button("この表現を適用", key=f"validation_apply_variant_{project.key}_{selected_id}"):
                active_block["summary"] = choice
                st.session_state[f"validation_summary_edit_{project.key}_{selected_id}"] = choice

    with st.expander("元のテキスト"):
        st.write(active_block.get("text", ""))

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
