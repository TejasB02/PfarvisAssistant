# =============================================================================
# module_rag/dates.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : the standalone question from pipeline.ask.
# Hands off: a predicate on a note's event_time (Person 1's parsed datetime),
#            used by pipeline.ask to add notes whose event falls in the time
#            span the question asks about.
# Why      : embeddings are weak at dates ("meetings in December" does not
#            match "review on Wednesday 9 December"), but every note already
#            has a parsed event_time. Deterministic, no LLM.
# =============================================================================
"""Detect a time span in a question ("in December", "this week") and match event times."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Callable, Optional

MONTHS = ("january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december")
_MONTH_RE = re.compile(
    r"\b(?:(january|february|march|april|june|july|august|september|october|november|december)"
    r"|(?<=\bin )(may)|(?<=\bof )(may))\b(?:\s+(\d{4}))?"
)

EventFilter = Callable[[datetime], bool]


def event_filter(query: str, now: Optional[datetime] = None) -> Optional[EventFilter]:
    """Return a test for event_time if ``query`` names a time span, else None.

    Spans: "today", "tomorrow", "this week", "next week", "this month",
    "next month", and month names ("December", "December 2026"; "may" only as
    "in May" / "of May"). A month without a year matches that month in any year.
    Weeks run Monday to Sunday.
    """
    text = " ".join((query or "").lower().split())
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def between(start: datetime, end: datetime) -> EventFilter:
        return lambda dt: start <= dt < end

    if re.search(r"\bnext 90 days\b", text):
        return between(now, now + timedelta(days=90))
    if re.search(r"\btoday\b|\btonight\b", text):
        return between(today, today + timedelta(days=1))
    if re.search(r"\btomorrow\b", text):
        return between(today + timedelta(days=1), today + timedelta(days=2))
    week_start = today - timedelta(days=today.weekday())
    if re.search(r"\bthis week\b", text):
        return between(week_start, week_start + timedelta(days=7))
    if re.search(r"\bnext week\b", text):
        return between(week_start + timedelta(days=7), week_start + timedelta(days=14))
    if re.search(r"\bthis month\b", text):
        return lambda dt: (dt.year, dt.month) == (now.year, now.month)
    if re.search(r"\bnext month\b", text):
        year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
        return lambda dt: (dt.year, dt.month) == (year, month)

    match = _MONTH_RE.search(text)
    if match:
        name = next(g for g in match.groups()[:3] if g)
        month = MONTHS.index(name) + 1
        year = int(match.group(4)) if match.group(4) else None
        return lambda dt: dt.month == month and (year is None or dt.year == year)
    return None
