# =============================================================================
# module_ingest/extract.py - owner: Person 1 (Ingestion & OCR)
# Receives : raw note text (and the note's own timestamp) from parser.parse_file.
# Hands off: client name, tag list and event_time back to parser.parse_file,
#            which puts them into the NoteItem dict.
#            client/tags drive Person 2's graph edges; event_time drives
#            Person 4's reminders.
# Rule     : deterministic (no LLM). Same text + same reference -> same output.
# STATUS   : STUB - returns empty values.
# =============================================================================
"""Deterministic metadata extraction: client, tags, event time."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


def guess_client(text: str) -> Optional[str]:
    """Guess which client a note is about.

    Inputs:
        text: full note text.
    Output:
        Canonical client name (e.g. "Quillmont Systems") or None.
    Suggested rules (deterministic, in order):
        1. An explicit "Client: <Name>" line.
        2. A known-client list (e.g. built from folder names under
           ".../Clients/<Name>/" during the scan) matched case-insensitively.
        3. Otherwise None. Never guess from vague words like "the clients".
    Edge cases:
        - Empty text -> None.
        - Two different clients mentioned -> return the first explicit one;
          Person 3's prompt handles ambiguity at question time.
    """
    # TODO(person 1): implement the rules above.
    return None


def extract_tags(text: str) -> list[str]:
    """Extract tags: #hashtags plus simple keyword rules.

    Inputs:
        text: full note text.
    Output:
        Sorted, de-duplicated, lowercase tags WITHOUT "#", e.g.
        ["budget", "q3-rollout"]. Empty list if none.
    Suggested rules:
        - Hashtags: regex like r"(?<!\\w)#([A-Za-z][\\w-]*)"; ignore Markdown
          headings ("# Title") and hex colours.
        - Keywords: e.g. "invoice"/"budget" -> "budget", "interview" -> "hiring".
          Keep the keyword map small and documented.
    Edge cases: empty text -> [].
    """
    # TODO(person 1): implement hashtag + keyword extraction.
    return []


def extract_event_time(text: str, reference: datetime) -> Optional[datetime]:
    """Find when the thing described in the note happens.

    Inputs:
        text: full note text, e.g. "Call with the clients at 4pm today".
        reference: the note's OWN timestamp (file modified time). Relative words
            ("today", "tomorrow", "4pm", "Friday") resolve against this, NOT
            against the current date, so results are reproducible.
    Output:
        Naive local datetime, or None if no time is found.
    Suggested approach:
        dateparser.search.search_dates(text, languages=["en"], settings={
            "RELATIVE_BASE": reference, "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": False})
        and take the first match that contains a time or a date word.
        Import dateparser INSIDE this function.
    Edge cases:
        - A bare time ("4pm") -> reference date at 16:00.
        - Several times -> the earliest one at or after the reference.
        - Ambiguous numbers ("Q3", "v2.1", phone numbers) must NOT become dates.
        - Empty text -> None.
    """
    # TODO(person 1): implement with dateparser as described above.
    return None
