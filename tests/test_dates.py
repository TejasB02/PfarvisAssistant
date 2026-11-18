"""Date check: questions naming a time span get the notes whose event falls in it."""

from datetime import datetime
from pathlib import Path

import pytest

from contracts import ContextItem
from module_rag import dates, engine, pipeline

NOW = datetime(2026, 11, 18, 15, 0)  # a Wednesday


@pytest.mark.parametrize("question, inside, outside", [
    ("Do I have any meetings in December?", datetime(2026, 12, 9, 14), datetime(2026, 11, 30, 9)),
    ("anything in December 2027?", datetime(2027, 12, 1, 9), datetime(2026, 12, 9, 14)),
    ("what's on today?", datetime(2026, 11, 18, 16), datetime(2026, 11, 19, 9)),
    ("meetings tomorrow", datetime(2026, 11, 19, 9), datetime(2026, 11, 18, 16)),
    ("what's due this week?", datetime(2026, 11, 22, 23), datetime(2026, 11, 23, 0)),
    ("next week's calls", datetime(2026, 11, 23, 10), datetime(2026, 11, 22, 10)),
    ("anything this month", datetime(2026, 11, 1, 8), datetime(2026, 12, 1, 8)),
    ("anything next month", datetime(2026, 12, 31, 8), datetime(2026, 11, 30, 8)),
    ("what happens in May?", datetime(2027, 5, 3, 9), datetime(2027, 6, 3, 9)),
])
def test_event_filter_spans(question, inside, outside):
    matches = dates.event_filter(question, now=NOW)
    assert matches is not None
    assert matches(inside) and not matches(outside)


@pytest.mark.parametrize("question", [
    "When is my call with the clients?", "May I see my to-dos?", "Which client?",
    "what did Dana say about the rollout", "",
])
def test_no_time_span_means_no_filter(question):
    assert dates.event_filter(question, now=NOW) is None


def _note(note_id, name, event_time):
    return {"id": note_id, "title": name, "content": f"content of {name}",
            "filepath": f"/notes/{name}", "timestamp": None, "client": None,
            "tags": [], "event_time": event_time}


class DatedStore:
    def __init__(self, notes, hits=()):
        self.notes, self.hits = notes, list(hits)

    def hybrid_search(self, query, k=5, hops=1):
        return [dict(h) for h in self.hits]

    def get_note(self, note_id):
        return next((n for n in self.notes if n["id"] == note_id), None)

    def all_notes(self):
        return [dict(n) for n in self.notes]


@pytest.fixture
def captured_context(monkeypatch):
    seen = []

    def fake_answer(query, retrieved_context, chat_history):
        seen.append(retrieved_context)
        return {"answer": "ok", "sources": [], "grounded": False}

    monkeypatch.setattr(engine, "answer_user_query", fake_answer)
    return seen


def test_ask_adds_notes_with_an_event_in_the_month_asked_about(captured_context):
    weak_hit = ContextItem(filepath="/notes/todo.txt", title="todo", content="x", score=0.3, via="semantic")
    store = DatedStore([
        _note("a", "december_review.md", datetime(2026, 12, 9, 14)),
        _note("b", "november_call.txt", datetime(2026, 11, 18, 16)),
        _note("c", "undated.md", None),
    ], hits=[weak_hit])
    pipeline.ask("Do I have any meetings in December?", [], store)
    context = captured_context[0]
    assert [Path(c["filepath"]).name for c in context] == ["december_review.md", "todo.txt"]
    assert context[0]["via"] == "semantic" and context[0]["score"] == 1.0


def test_dated_note_also_found_by_search_is_not_duplicated(captured_context):
    note = _note("a", "december_review.md", datetime(2026, 12, 9, 14))
    hit = ContextItem(filepath=note["filepath"], title="t", content="c", score=0.4, via="semantic")
    pipeline.ask("meetings in December", [], DatedStore([note], hits=[hit]))
    assert [c["filepath"] for c in captured_context[0]] == ["/notes/december_review.md"]


def test_no_time_span_does_not_read_all_notes(captured_context):
    class NoListing(DatedStore):
        def all_notes(self):
            raise AssertionError("must not list notes without a time span")
    pipeline.ask("When is my call with the clients?", [], NoListing([]))


def test_store_without_all_notes_still_answers(captured_context):
    class BareStore:
        def hybrid_search(self, query, k=5, hops=1):
            return []
        def get_note(self, note_id):
            return None
    pipeline.ask("anything in December?", [], BareStore())
    assert captured_context[0] == []
