# =============================================================================
# contracts.py - THE shared data contracts (owned by ALL FOUR)
# Changing anything here needs agreement from all four people, because every
# module depends on these shapes.
#
# Person 1 (ingest)  -> produces note dicts  == asdict(NoteItem)
# Person 2 (storage) -> consumes note dicts, produces ContextItem lists
# Person 3 (rag)     -> consumes ContextItem lists, produces AnswerDict
# Person 4 (ui)      -> consumes note dicts + AnswerDict
# =============================================================================
"""Shared dataclasses, typed dicts and helpers used across all modules.

Conventions every module relies on:
- ``filepath`` is the ABSOLUTE, resolved path of the source file, as a string.
- ``id`` is ``make_note_id(filepath)``: sha256 hex of that absolute path.
- All datetimes are NAIVE LOCAL time (``datetime.now()`` style, no tzinfo).
- ``tags`` are lowercase strings WITHOUT the leading ``#`` (e.g. ``"q3-rollout"``).
- A note dict is exactly ``asdict(NoteItem)``; datetimes stay ``datetime`` objects.
"""

import hashlib
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional, TypedDict, Union

FALLBACK_MESSAGE = "I couldn't find that in your notes."


@dataclass
class NoteItem:
    """One ingested note (one file). Field order is part of the contract."""

    id: str                                   # make_note_id(filepath)
    title: str
    content: str
    filepath: str                             # absolute path as str
    timestamp: Optional[datetime] = None      # file modified time
    client: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    event_time: Optional[datetime] = None     # when the thing in the note happens


# Field names in contract order; handy for schema checks.
NOTE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(NoteItem))
_DATETIME_FIELDS = ("timestamp", "event_time")


class ContextItem(TypedDict):
    """One retrieved note handed from storage (Person 2) to RAG (Person 3)."""

    filepath: str
    title: str
    content: str
    score: float                              # similarity, 0..1 (higher = closer)
    via: Literal["semantic", "graph"]         # how it was found


class AnswerDict(TypedDict):
    """The answer handed from RAG (Person 3) to the UI (Person 4)."""

    answer: str
    sources: List[str]                        # filepaths actually cited
    grounded: bool                            # True only if backed by cited notes


def make_note_id(filepath: Union[str, Path]) -> str:
    """Return the sha256 hex digest of the absolute, resolved filepath.

    Deterministic: the same file always gets the same id, whether given as a
    relative or absolute path.
    """
    absolute = str(Path(filepath).expanduser().resolve())
    return hashlib.sha256(absolute.encode("utf-8")).hexdigest()


def note_to_dict(note: NoteItem) -> dict[str, Any]:
    """Convert a NoteItem to the shared note dict (exactly ``asdict``)."""
    return asdict(note)


def note_from_dict(d: dict[str, Any]) -> NoteItem:
    """Build a NoteItem from a note dict.

    Unknown keys are ignored. Datetime fields may be ``datetime`` objects or ISO
    strings (as stored in Chroma metadata); empty strings become ``None``.
    Missing required keys (id, title, content, filepath) raise ``KeyError``.
    """
    known = {name: d[name] for name in NOTE_FIELDS if name in d}
    for name in _DATETIME_FIELDS:
        value = known.get(name)
        if isinstance(value, str):
            known[name] = datetime.fromisoformat(value) if value else None
    known["tags"] = list(known.get("tags") or [])
    for required in ("id", "title", "content", "filepath"):
        if required not in known:
            raise KeyError(f"note dict is missing required key {required!r}")
    return NoteItem(**known)


def fallback_answer() -> AnswerDict:
    """Return a fresh copy of the standard 'not in your notes' answer."""
    return {"answer": FALLBACK_MESSAGE, "sources": [], "grounded": False}
