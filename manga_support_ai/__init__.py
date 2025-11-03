"""Core package for the Manga Support AI demo application."""

from . import config
from .models import EntryRecord, ProjectData

__all__ = [
    "config",
    "EntryRecord",
    "ProjectData",
]

