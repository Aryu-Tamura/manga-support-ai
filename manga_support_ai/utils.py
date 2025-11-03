"""Utility helpers shared across the application."""

import io
import json
import logging
import re
from typing import Any, Optional

import streamlit as st

from epub_utils import extract_text_from_epub


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_json_from_string(raw: str) -> Optional[Any]:
    if raw is None:
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.sub(r"^```[a-zA-Z]*\n", "", text)
    fenced = re.sub(r"\n```$", "", fenced)
    try:
        return json.loads(fenced)
    except Exception:
        pass

    match = re.search(r"\[\s*{.*}\s*]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def read_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = (uploaded_file.name or "").lower()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    raw_bytes = uploaded_file.read()
    if name.endswith(".epub"):
        return extract_text_from_epub(raw_bytes)
    for encoding in ("utf-8", "utf-16", "shift_jis", "cp932"):
        try:
            return raw_bytes.decode(encoding)
        except Exception:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def create_docx_bytes(document_text: str) -> Optional[bytes]:
    if not document_text.strip():
        return None
    try:
        from docx import Document
    except ImportError:  # pragma: no cover
        st.info("Word形式でのダウンロードには `python-docx` のインストールが必要です。")
        return None

    doc = Document()
    for line in document_text.splitlines():
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()
