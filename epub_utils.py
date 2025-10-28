# -*- coding: utf-8 -*-
"""EPUB関連のユーティリティ"""

import os
import tempfile
from typing import List

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def extract_text_from_epub(file_bytes: bytes) -> str:
    """EPUBバイト列から本文テキストを抽出して連結する"""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        temp_path = tmp_path
        book = epub.read_epub(tmp_path)
        chapters: List[str] = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    chapters.append(text)
        return "\n\n".join(chapters).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

