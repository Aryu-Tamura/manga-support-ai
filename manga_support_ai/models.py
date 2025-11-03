"""Data models and helpers for project content."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from .llm_workflow import Panel  # noqa: F401


@dataclass
class EntryRecord:
    id: int
    text: str
    type: str
    speakers: List[str]
    time: str
    location: str
    tone: str
    emotion: str
    action: str
    entities: List[str]
    source_span: Dict[str, int]
    summary: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, object], *, fallback_id: int) -> "EntryRecord":
        entry_id = _coerce_entry_id(data.get("id"), fallback_id)
        summary = str(data.get("summary") or "")
        return cls(
            id=entry_id,
            text=str(data.get("text") or ""),
            type=str(data.get("type") or "unknown"),
            speakers=[str(s) for s in (data.get("speakers") or [])],
            time=str(data.get("time") or "unknown"),
            location=str(data.get("location") or ""),
            tone=str(data.get("tone") or "neutral"),
            emotion=str(data.get("emotion") or "neutral"),
            action=str(data.get("action") or ""),
            entities=[str(e) for e in (data.get("entities") or [])],
            source_span=data.get("source_local_span")
            or data.get("source_span")
            or {"start": -1, "end": -1},
            summary=summary,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type,
            "speakers": self.speakers,
            "time": self.time,
            "location": self.location,
            "tone": self.tone,
            "emotion": self.emotion,
            "action": self.action,
            "entities": self.entities,
            "source_local_span": self.source_span,
            "summary": self.summary,
        }


@dataclass
class ProjectData:
    key: str
    title: str
    summary: str
    entries: List[EntryRecord]
    characters: List[Dict[str, str]]
    full_text: str = ""
    source_path: Optional[Path] = None

    @property
    def chunk_count(self) -> int:
        return len(self.entries)

    def __post_init__(self) -> None:
        if not self.full_text and self.entries:
            self.full_text = "\n\n".join(entry.text for entry in self.entries)


def entries_to_context(
    entries: Iterable[EntryRecord],
    limit_chars: Optional[int] = None,
) -> str:
    joined = "\n\n".join(f"[{entry.id}] {entry.text}" for entry in entries)
    if limit_chars is None:
        return joined
    return joined[:limit_chars]


def get_entry_slice(project: ProjectData, start: int, end: int) -> List[EntryRecord]:
    if project.chunk_count == 0:
        return []
    start_idx = max(1, start)
    end_idx = min(project.chunk_count, end)
    if start_idx > end_idx:
        return []
    return project.entries[start_idx - 1 : end_idx]


def find_character_contexts(
    project: ProjectData,
    name: str,
    limit: Optional[int] = None,
) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    target = name.strip()
    if not target:
        return hits
    for entry in project.entries:
        if target in entry.text:
            snippet = entry.text.strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            hits.append((entry.id, snippet))
        if limit is not None and len(hits) >= limit:
            break
    return hits


def panels_to_entries(
    panels: Iterable["Panel"],
    canonical_names: Optional[List[str]] = None,
) -> List[EntryRecord]:
    """Convert legacy Panel objects into EntryRecord instances."""
    name_map = _build_name_map(canonical_names or [])
    entries: List[EntryRecord] = []
    for idx, panel in enumerate(panels, start=1):
        raw_speaker = getattr(panel, "speaker", "")
        raw_list = []
        panel_speakers = getattr(panel, "speakers", None)
        if isinstance(panel_speakers, (list, tuple)):
            raw_list.extend(panel_speakers)
        elif isinstance(panel_speakers, str) and panel_speakers.strip():
            raw_list.append(panel_speakers)
        if not raw_list and raw_speaker:
            raw_list.append(raw_speaker)

        speakers: List[str] = []
        seen = set()
        for candidate in raw_list:
            name = str(candidate or "").strip()
            if not name:
                continue
            normalised = _normalize_speaker(name, name_map)
            if not normalised:
                continue
            key = normalised.lower()
            if key in seen:
                continue
            seen.add(key)
            speakers.append(normalised)

        entries.append(
            EntryRecord(
                id=idx,
                text=panel.text,
                type=panel.type,
                speakers=speakers,
                time=panel.time,
                location=panel.location or getattr(panel, "scene", ""),
                tone=panel.tone,
                emotion=panel.emotion,
                action=panel.action,
                entities=list(panel.entities or []),
                source_span=panel.source_span or {"start": -1, "end": -1},
                summary="",
            )
        )
    return entries


def _coerce_entry_id(value: object, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = re.sub(r"\D", "", value)
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
    return fallback


def _build_name_map(names: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name in names:
        if not name:
            continue
        norm = _normalize_token(name)
        if norm:
            mapping[norm] = name
    return mapping


def _normalize_speaker(speaker: str, name_map: Dict[str, str]) -> str:
    norm = _normalize_token(speaker)
    if norm in name_map:
        return name_map[norm]
    for candidate_norm, canonical in name_map.items():
        if candidate_norm in norm or norm in candidate_norm:
            return canonical
    return speaker.strip()


def _normalize_token(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()
