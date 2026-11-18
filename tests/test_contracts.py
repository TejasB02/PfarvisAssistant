"""The shared contract must not drift: field names, order, defaults, ids, dicts."""

from dataclasses import MISSING, asdict, fields
from datetime import datetime

from contracts import (
    FALLBACK_MESSAGE,
    NOTE_FIELDS,
    NoteItem,
    fallback_answer,
    make_note_id,
    note_from_dict,
    note_to_dict,
)

EXPECTED_FIELDS = ["id", "title", "content", "filepath", "timestamp", "client", "tags", "event_time"]


def _sample_note(**overrides) -> NoteItem:
    values = dict(
        id=make_note_id("sample_notes/meetings/2026-11/call_prep.txt"),
        title="Call prep",
        content="Call with the clients at 4pm today.",
        filepath="sample_notes/meetings/2026-11/call_prep.txt",
        timestamp=datetime(2026, 11, 18, 9, 30),
        client="Quillmont Systems",
        tags=["q3-rollout", "client-call"],
        event_time=datetime(2026, 11, 18, 16, 0),
    )
    values.update(overrides)
    return NoteItem(**values)


def test_noteitem_field_names_and_order():
    assert [f.name for f in fields(NoteItem)] == EXPECTED_FIELDS
    assert list(NOTE_FIELDS) == EXPECTED_FIELDS


def test_noteitem_defaults():
    by_name = {f.name: f for f in fields(NoteItem)}
    for required in ("id", "title", "content", "filepath"):
        assert by_name[required].default is MISSING
        assert by_name[required].default_factory is MISSING
    assert by_name["timestamp"].default is None
    assert by_name["client"].default is None
    assert by_name["event_time"].default is None
    assert by_name["tags"].default_factory is list

    a = NoteItem(id="a", title="t", content="c", filepath="f")
    b = NoteItem(id="b", title="t", content="c", filepath="f")
    assert a.tags == [] and a.timestamp is None and a.client is None and a.event_time is None
    a.tags.append("x")
    assert b.tags == [], "tags default must not be shared between instances"


def test_make_note_id_is_deterministic(tmp_path):
    path = tmp_path / "note.md"
    first = make_note_id(path)
    assert first == make_note_id(path)
    assert first == make_note_id(str(path))
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)
    assert first != make_note_id(tmp_path / "other.md")


def test_make_note_id_relative_equals_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert make_note_id("note.md") == make_note_id(tmp_path / "note.md")


def test_dict_round_trip_keeps_datetimes():
    note = _sample_note()
    d = note_to_dict(note)
    assert d == asdict(note)
    assert list(d) == EXPECTED_FIELDS
    assert isinstance(d["timestamp"], datetime) and isinstance(d["event_time"], datetime)
    assert note_from_dict(d) == note


def test_round_trip_with_defaults_and_iso_strings():
    bare = NoteItem(id="x", title="t", content="c", filepath="f")
    assert note_from_dict(note_to_dict(bare)) == bare

    note = _sample_note()
    d = note_to_dict(note)
    d["timestamp"] = d["timestamp"].isoformat()
    d["event_time"] = d["event_time"].isoformat()
    d["unexpected_extra_key"] = "ignored"
    assert note_from_dict(d) == note


def test_fallback_answer_shape():
    fb = fallback_answer()
    assert fb == {"answer": FALLBACK_MESSAGE, "sources": [], "grounded": False}
    fb["sources"].append("mutated")
    assert fallback_answer()["sources"] == []
    assert FALLBACK_MESSAGE == "I couldn't find that in your notes."
