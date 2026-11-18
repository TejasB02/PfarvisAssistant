# =============================================================================
# module_ingest/parser.py - owner: Person 1 (Ingestion & OCR)
# Receives : a directory path (config.NOTE_DIR) from app.py.
# Hands off: list[dict], one per file, each exactly asdict(contracts.NoteItem)
#            -> Person 2 (NotesStore.add) and Person 4 (sidebar + reminders).
# Uses     : ocr.image_to_text for images, extract.* for client/tags/event_time.
# STATUS   : STUB - returns hardcoded mock notes that mirror sample_notes/.
# =============================================================================
"""Recursive scan of a notes folder and per-file parsing."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import config
from contracts import NoteItem, make_note_id, note_to_dict

log = logging.getLogger(__name__)


def scan_and_parse(dir_path: str | Path) -> list[dict]:
    """Recursively scan ``dir_path`` and return one note dict per readable file.

    THE Person 1 deliverable.

    Inputs:
        dir_path: folder to scan (relative or absolute). Sub-folders included.
    Output:
        list of dicts, each exactly ``asdict(NoteItem)``: datetimes stay
        ``datetime`` objects, ``filepath`` is absolute, ``id`` is
        ``make_note_id(filepath)``. Order: sorted by filepath (deterministic).
    Edge cases:
        - Missing folder -> log a warning, return [].
        - Files with unsupported extensions, hidden files (".*") and files that
          ``parse_file`` returns None for are skipped, never crash the scan.
        - Duplicate content in two files is kept (two different notes).
    """
    # TODO(person 1): replace the mock with the real scan:
    #   root = Path(dir_path).expanduser().resolve()
    #   for path in sorted(root.rglob("*")):
    #       if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
    #           note = parse_file(path)   # wrap in try/except, log and continue
    #           if note: notes.append(note)
    root = Path(dir_path).expanduser().resolve()
    if not root.exists():
        log.warning("Notes folder not found: %s", root)
        return []
    return _mock_notes(root)


def parse_file(path: Path) -> Optional[dict]:
    """Parse one file into a note dict, dispatching on its extension.

    Inputs:
        path: a file path. Supported: .txt .md .pdf .docx .png .jpg .jpeg
              (config.SUPPORTED_EXTENSIONS, case-insensitive).
    Output:
        ``asdict(NoteItem)`` or None.
        - title: first Markdown heading / first non-empty line / file stem.
        - timestamp: file modified time (naive local datetime).
        - client: extract.guess_client(content), or the folder name when the
          file lives under ".../Clients/<Name>/".
        - tags: extract.extract_tags(content).
        - event_time: extract.extract_event_time(content, reference=timestamp).
    Edge cases:
        - Unsupported extension, unreadable, encrypted or empty file -> log and
          return None (never raise).
        - Text encodings: try utf-8, fall back to utf-8 with errors="replace".
        - .pdf via pypdf, .docx via python-docx, images via ocr.image_to_text.
          Import those libraries INSIDE the branch that needs them.
    """
    # TODO(person 1): implement the real dispatch described above.
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        log.info("Skipping unsupported or missing file: %s", path)
        return None
    resolved = path.resolve()
    note = NoteItem(
        id=make_note_id(resolved),
        title=path.stem,
        content=f"(mock) Parsed content of {path.name} would appear here.",
        filepath=str(resolved),
        timestamp=datetime.fromtimestamp(path.stat().st_mtime),
    )
    return note_to_dict(note)


# --- Mock data (delete once the real scan works) -------------------------------

def _mock_notes(root: Path) -> list[dict]:
    """Hardcoded notes shaped like the real sample_notes/ files.

    The first note's event_time is 5 minutes from now so the reminder flow
    (notifier -> toast -> "Add to chat") can be demoed immediately.
    """
    now = datetime.now().replace(second=0, microsecond=0)
    specs = [
        dict(
            rel="meetings/2026-11/call_prep.txt",
            title="Call prep - 4pm client call",
            content=(
                "Call with the clients at 4pm today - rollout check-in.\n"
                "Agenda: wave 2 timeline, support hours over the holidays, pricing tier.\n"
                "#q3-rollout #client-call"
            ),
            client=None,
            tags=["q3-rollout", "client-call"],
            event_time=now + timedelta(minutes=5),  # mock: fires a reminder right away
        ),
        dict(
            rel="Work/Clients/Quillmont/account_overview.md",
            title="Quillmont Systems - account overview",
            content=(
                "Client: Quillmont Systems. Main contacts: Dana Okafor (ops lead) and "
                "Lukas Brandt (procurement). We run the rollout check-in calls with "
                "Quillmont's ops team. #quillmont #q3-rollout"
            ),
            client="Quillmont Systems",
            tags=["quillmont", "q3-rollout"],
            event_time=None,
        ),
        dict(
            rel="Desktop dump/untitled 3.txt",
            title="untitled 3",
            content=(
                "re the 4pm rollout check-in: Dana confirmed it's virtual, not in person "
                "this time. Video link is in the calendar invite. #q3-rollout"
            ),
            client=None,
            tags=["q3-rollout"],
            event_time=None,
        ),
        dict(
            rel="old_notes/2026/q2_review.md",
            title="Q2 review - Quillmont",
            content=(
                "Quillmont Systems Q2 review: on budget, two late deliveries, "
                "they want weekly status emails. #quillmont #budget"
            ),
            client="Quillmont Systems",
            tags=["quillmont", "budget"],
            event_time=None,
        ),
        dict(
            rel="Work/Clients/Norrbridge/kickoff.md",
            title="Norrbridge Foods - pilot kickoff",
            content=(
                "Client: Norrbridge Foods. Pilot kickoff is in person at their Easton "
                "office on Friday at 10am. Contact: Ines Valdier. #norrbridge #pilot"
            ),
            client="Norrbridge Foods",
            tags=["norrbridge", "pilot"],
            event_time=(now + timedelta(days=2)).replace(hour=10, minute=0),
        ),
    ]
    notes: list[dict] = []
    for spec in specs:
        filepath = str((root / spec["rel"]).resolve())
        note = NoteItem(
            id=make_note_id(filepath),
            title=spec["title"],
            content=spec["content"],
            filepath=filepath,
            timestamp=now,
            client=spec["client"],
            tags=list(spec["tags"]),
            event_time=spec["event_time"],
        )
        notes.append(note_to_dict(note))
    return sorted(notes, key=lambda n: n["filepath"])
