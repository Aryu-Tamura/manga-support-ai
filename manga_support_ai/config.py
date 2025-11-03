"""Application configuration constants and defaults."""

from pathlib import Path
from typing import Optional, Set

DATA_DIR = Path("data")
DEFAULT_MODEL = "gpt-4o-mini"
SUMMARY_GRAIN_OPTIONS = list(range(50, 801, 50))
MAX_CONTEXT_CHARS: Optional[int] = None
SAMPLE_ONLY_PROJECT_KEYS: Set[str] = {"project1", "project2"}

BASE_PROJECT_DEFINITIONS = [
    {
        "key": "project1",
        "title": "プロジェクト1：銀河鉄道の夜",
        "panel_file": DATA_DIR / "gingatetudono_yoru_labeled.json",
        "character_file": DATA_DIR / "character_gingatetudonoyoru.json",
    },
    {
        "key": "project2",
        "title": "プロジェクト2：井上尚弥の書籍",
        "panel_file": DATA_DIR / "inouenaoya_labeled.json",
        "character_file": DATA_DIR / "character_inouenaoya.json",
    },
]

PROJECT_INDEX_FILE = DATA_DIR / "projects_index.json"
