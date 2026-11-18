"""Unit and integration tests for module_ingest (parser, ocr, extract)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from contracts import NOTE_FIELDS, make_note_id
from module_ingest import extract, ocr, parser

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_notes"


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def test_parse_txt_and_md_with_relative_event_time(tmp_path):
    note_file = tmp_path / "call_prep.txt"
    note_file.write_text(
        "Call with the clients at 4pm today - rollout check-in.\n#q3-rollout #Client-Call\n",
        encoding="utf-8",
    )
    ref = datetime(2026, 11, 18, 9, 15)
    _set_mtime(note_file, ref)

    parsed = parser.parse_file(note_file)
    assert parsed is not None
    assert list(parsed) == list(NOTE_FIELDS)
    assert parsed["id"] == make_note_id(note_file)
    assert parsed["title"] == "Call with the clients at 4pm today - rollout check-in."
    assert parsed["timestamp"] == ref
    assert parsed["event_time"] == datetime(2026, 11, 18, 16, 0)
    assert parsed["tags"] == ["client-call", "q3-rollout"]
    assert parsed["client"] is None


def test_markdown_heading_title_and_client_folder_fallback(tmp_path):
    client_dir = tmp_path / "Work" / "Clients" / "Quillmont"
    client_dir.mkdir(parents=True)
    md_file = client_dir / "notes.md"
    md_file.write_text("# Weekly Sync\nDiscussed Q3 rollout goals and v2.1 release.\n", encoding="utf-8")

    parsed = parser.parse_file(md_file)
    assert parsed is not None
    assert parsed["title"] == "Weekly Sync"
    assert parsed["client"] == "Quillmont"
    assert parsed["event_time"] is None  # Q3 and v2.1 must not trigger dates
    assert parsed["tags"] == ["q3-rollout"]


def test_explicit_client_overrides_folder_name(tmp_path):
    client_dir = tmp_path / "Work" / "Clients" / "Quillmont"
    client_dir.mkdir(parents=True)
    md_file = client_dir / "overview.md"
    md_file.write_text("Client: Quillmont Systems.\nOps check-in.\n", encoding="utf-8")

    parsed = parser.parse_file(md_file)
    assert parsed is not None
    assert parsed["client"] == "Quillmont Systems"


def test_parse_docx(tmp_path):
    import docx

    doc_path = tmp_path / "meeting.docx"
    doc = docx.Document()
    doc.add_paragraph("Quarterly Budget Review")
    doc.add_paragraph("Client: Norrbridge Foods")
    doc.add_paragraph("Review invoice schedule #finance")
    doc.save(str(doc_path))

    parsed = parser.parse_file(doc_path)
    assert parsed is not None
    assert parsed["title"] == "Quarterly Budget Review"
    assert parsed["client"] == "Norrbridge Foods"
    assert parsed["tags"] == ["budget", "finance"]


def test_parse_pdf_and_graceful_failure(tmp_path, monkeypatch):
    # Verify corrupt PDF does not crash and returns None
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 corrupted bytes")
    assert parser.parse_file(bad_pdf) is None

    # Verify valid PDF path via mocked PdfReader
    good_pdf = tmp_path / "brief.pdf"
    good_pdf.write_bytes(b"%PDF-1.4 placeholder")

    class FakePage:
        def extract_text(self):
            return "Executive Brief\nClient: Acme Corp\n#strategy"

    class FakeReader:
        is_encrypted = False
        pages = [FakePage()]

    import pypdf
    monkeypatch.setattr("pypdf.PdfReader", lambda _: FakeReader())

    parsed = parser.parse_file(good_pdf)
    assert parsed is not None
    assert parsed["title"] == "Executive Brief"
    assert parsed["client"] == "Acme Corp"
    assert parsed["tags"] == ["strategy"]


def test_ocr_graceful_failures_and_image_parse(tmp_path, monkeypatch):
    corrupt_img = tmp_path / "broken.png"
    corrupt_img.write_bytes(b"not-a-valid-png")
    assert ocr.image_to_text(corrupt_img) == ""
    assert parser.parse_file(corrupt_img) is None

    from PIL import Image

    img_path = tmp_path / "board.png"
    Image.new("RGB", (60, 20), color="white").save(img_path)

    monkeypatch.setattr(ocr, "image_to_text", lambda p: "Whiteboard Action Items\n#offsite")
    parsed = parser.parse_file(img_path)
    assert parsed is not None
    assert parsed["title"] == "Whiteboard Action Items"
    assert parsed["tags"] == ["offsite"]


def test_scan_and_parse_skips_hidden_and_empty_files(tmp_path):
    (tmp_path / ".Rhistory").write_text("ls()\n", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n\t  ", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    (tmp_path / "valid.txt").write_text("Hello world\n", encoding="utf-8")

    results = parser.scan_and_parse(tmp_path)
    assert len(results) == 1
    assert results[0]["title"] == "Hello world"


def test_scan_sample_notes_directory():
    notes = parser.scan_and_parse(SAMPLE_DIR)
    filenames = {Path(n["filepath"]).name for n in notes}

    # Hidden .Rhistory files must be excluded; all text/md sample notes must be present
    assert ".Rhistory" not in filenames
    expected_text_files = {
        "todo.txt",
        "untitled 3.txt",
        "kickoff.md",
        "account_overview.md",
        "1on1_mei.md",
        "call_prep.txt",
        "offsite_planning.md",
        "q2_review.md",
        "ideas.txt",
    }
    assert expected_text_files.issubset(filenames)

    by_name = {Path(n["filepath"]).name: n for n in notes}
    call_note = by_name["call_prep.txt"]
    assert call_note["event_time"] is not None
    assert call_note["event_time"].hour == 16
    assert call_note["event_time"].minute == 0
    assert call_note["event_time"].date() == call_note["timestamp"].date()
    assert "q3-rollout" in call_note["tags"]

    assert by_name["account_overview.md"]["client"] == "Quillmont Systems"
    assert by_name["kickoff.md"]["client"] == "Norrbridge Foods"

def test_watcher_debounces_changes_and_handles_deletes(tmp_path):
    import threading
    import time
    from module_ingest import watcher

    changed: list[dict] = []
    deleted: list[str] = []
    change_event = threading.Event()
    delete_event = threading.Event()

    def on_change(note: dict) -> None:
        changed.append(note)
        change_event.set()
        raise RuntimeError("simulated callback error must not kill observer")

    def on_delete(note_id: str) -> None:
        deleted.append(note_id)
        delete_event.set()

    watcher.watch(tmp_path, on_change=on_change, on_delete=on_delete)
    time.sleep(0.2)

    # Temp and unsupported files must be ignored
    (tmp_path / "~$temp.docx").write_text("temp", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("a,b", encoding="utf-8")

    # Rapid writes to a supported file should debounce into a single on_change call
    note_file = tmp_path / "live_note.md"
    note_file.write_text("# Draft 1\n", encoding="utf-8")
    time.sleep(0.1)
    note_file.write_text("# Final Draft\n#budget\n", encoding="utf-8")

    assert change_event.wait(timeout=3.0), "timed out waiting for on_change"
    time.sleep(0.2)
    assert len(changed) == 1
    assert changed[0]["title"] == "Final Draft"
    assert changed[0]["tags"] == ["budget"]

    # Deleting the file should trigger on_delete with its deterministic note_id
    expected_id = make_note_id(note_file)
    note_file.unlink()
    assert delete_event.wait(timeout=3.0), "timed out waiting for on_delete"
    assert deleted == [expected_id]