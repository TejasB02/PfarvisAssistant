# =============================================================================
# module_ingest/extract.py - owner: Person 1 (Ingestion & OCR)
# Receives : raw note text (and the note's own timestamp) from parser.parse_file.
# Hands off: client name, tag list and event_time back to parser.parse_file,
#            which puts them into the NoteItem dict.
#            client/tags drive Person 2's graph edges; event_time drives
#            Person 4's reminders.
# Rule     : deterministic (no LLM). Same text + same reference -> same output.
# =============================================================================
"""Deterministic metadata extraction: client, tags, event time."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)

# Explicit "Client: <Name>" at start of a line; stops before period/newline.
_CLIENT_LINE_RE = re.compile(
    r"(?im)^\s*client\s*:\s*([A-Za-z0-9][A-Za-z0-9 &.,'-]*?)(?:\s*\.\s*(?:\s|$)|\s*$)"
)

# Hashtag regex: must not be preceded by a word char or '#', and must start with a letter
# (excludes Markdown headings "# Title" and hex colour codes "#1a2b3c").
_HASHTAG_RE = re.compile(r"(?<![\w#])#([A-Za-z][\w-]*)")

# Small, documented keyword -> canonical tag mapping.
KEYWORD_TAG_MAP: dict[str, str] = {
    "budget": "budget",
    "invoice": "budget",
    "hiring": "hiring",
    "interview": "hiring",
    "offsite": "offsite",
    "rollout": "q3-rollout",
}

# Tokens that indicate a genuine date or time expression (guards against "Q3", "v2.1", etc.).
_DATE_TIME_SIGNAL_RE = re.compile(
    r"(?i)\b(?:"
    r"today|tomorrow|yesterday|tonight|morning|afternoon|evening|"
    r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"next|this|am|pm|noon|midnight"
    r")\b|\b\d{1,2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
)

# Strings that dateparser might misinterpret as dates (e.g. quarters or software versions).
_FALSE_POSITIVE_RE = re.compile(r"(?i)^(?:q[1-4]|v?\d+\.\d+(?:\.\d+)*)$")

_KNOWN_CLIENTS: dict[str, str] = {}


def register_known_clients(root: Path) -> None:
    """Discover canonical client names from ``.../Clients/<Name>/`` folders under ``root``."""
    if not root.exists():
        return
    for clients_dir in sorted(root.rglob("Clients")):
        if clients_dir.is_dir():
            for child in sorted(clients_dir.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    _KNOWN_CLIENTS.setdefault(child.name.lower(), child.name)


def _get_known_clients() -> dict[str, str]:
    if not _KNOWN_CLIENTS:
        register_known_clients(Path(config.NOTE_DIR))
    return _KNOWN_CLIENTS


def guess_client(text: str) -> Optional[str]:
    """Guess which client a note is about.

    Inputs:
        text: full note text.
    Output:
        Canonical client name (e.g. "Quillmont Systems") or None.
    Rules (deterministic, in order):
        1. An explicit "Client: <Name>" line.
        2. A known-client list built from folder names under ".../Clients/<Name>/"
           matched case-insensitively.
        3. Otherwise None. Never guess from vague words like "the clients".
    """
    if not text or not text.strip():
        return None

    match = _CLIENT_LINE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            # Record canonical full name if it expands a folder shortname (e.g. Quillmont -> Quillmont Systems)
            for key, existing in list(_KNOWN_CLIENTS.items()):
                if candidate.lower().startswith(key) and len(candidate) > len(existing):
                    _KNOWN_CLIENTS[key] = candidate
            return candidate

    lowered = text.lower()
    for key, canonical in sorted(_get_known_clients().items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return canonical

    return None


def extract_tags(text: str) -> list[str]:
    """Extract tags: #hashtags plus simple keyword rules.

    Inputs:
        text: full note text.
    Output:
        Sorted, de-duplicated, lowercase tags WITHOUT "#", e.g.
        ["budget", "q3-rollout"]. Empty list if none.
    """
    if not text or not text.strip():
        return []

    tags: set[str] = set()

    for raw_tag in _HASHTAG_RE.findall(text):
        tags.add(raw_tag.lower())

    lowered = text.lower()
    for keyword, canonical_tag in KEYWORD_TAG_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}s?\b", lowered):
            tags.add(canonical_tag)

    return sorted(tags)


def extract_event_time(text: str, reference: datetime) -> Optional[datetime]:
    """Find when the thing described in the note happens.

    Inputs:
        text: full note text, e.g. "Call with the clients at 4pm today".
        reference: the note's OWN timestamp (file modified time). Relative words
            ("today", "tomorrow", "4pm", "Friday") resolve against this, NOT
            against the current date, so results are reproducible.
    Output:
        Naive local datetime, or None if no time is found.
    """
    if not text or not text.strip():
        return None

    from dateparser.search import search_dates

    # Ensure reference is naive local datetime
    ref = reference.replace(tzinfo=None) if reference.tzinfo else reference

    matches = search_dates(
        text,
        languages=["en"],
        settings={
            "RELATIVE_BASE": ref,
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if not matches:
        return None

    valid_dates: list[datetime] = []
    for matched_str, parsed_dt in matches:
        cleaned = matched_str.strip()
        if _FALSE_POSITIVE_RE.match(cleaned):
            continue
        if not _DATE_TIME_SIGNAL_RE.search(cleaned):
            continue
        dt_naive = parsed_dt.replace(tzinfo=None) if parsed_dt.tzinfo else parsed_dt
        valid_dates.append(dt_naive)

    if not valid_dates:
        return None

    # Prefer the earliest event at or after the note's reference timestamp;
    # fall back to the first valid parsed date if all mentioned times precede reference.
    upcoming = [dt for dt in valid_dates if dt >= ref]
    return min(upcoming) if upcoming else valid_dates[0]