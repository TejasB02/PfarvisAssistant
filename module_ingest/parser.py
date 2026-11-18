# =============================================================================
# module_ingest/parser.py - owner: Person 1 (Ingestion & OCR)
# Receives : a directory path (config.NOTE_DIR) from app.py.
# Hands off: list[dict], one per file, each exactly asdict(contracts.NoteItem)
#            -> Person 2 (NotesStore.add) and Person 4 (sidebar + reminders).
# Uses     : ocr.image_to_text for images, extract.* for client/tags/event_time.
# =============================================================================
"""Recursive scan of a notes folder and per-file parsing."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from contracts import NoteItem, make_note_id, note_to_dict
from module_ingest import extract, ocr

log = logging.getLogger(__name__)


def _is_hidden(path: Path, root: Optional[Path] = None) -> bool:
    """Return True if any path component (relative to root, or filename) starts with '.'."""
    target = path.relative_to(root) if root and path.is_relative_to(root) else Path(path.name)
    return any(part.startswith(".") for part in target.parts)


def _extract_title(content: str, fallback_stem: str) -> str:
    """Derive title from the first Markdown heading, first non-empty line, or file stem."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading
        return stripped[:120]
    return fallback_stem


def _client_from_path(path: Path) -> Optional[str]:
    """Return <Name> if ``path`` lives under a ``.../Clients/<Name>/...`` directory."""
    parts = path.parts
    for idx, part in enumerate(parts[:-1]):
        if part.lower() == "clients" and idx + 1 < len(parts) - 1:
            folder_name = parts[idx + 1]
            if folder_name and not folder_name.startswith("."):
                return folder_name
    return None


def _read_text_by_extension(path: Path) -> str:
    """Read raw text from ``path`` according to its lowercase suffix."""
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            log.warning("Skipping encrypted PDF: %s", path)
            return ""
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if suffix == ".docx":
        import docx

        document = docx.Document(str(path))
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return "\n".join(paragraphs).strip()

    if suffix in {".png", ".jpg", ".jpeg"}:
        return ocr.image_to_text(path)

    return ""


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
    root = Path(dir_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        log.warning("Notes folder not found: %s", root)
        return []

    extract.register_known_clients(root)

    notes: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_hidden(path, root):
            continue
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        try:
            note = parse_file(path)
            if note is not None:
                notes.append(note)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to parse %s: %s", path, exc)

    return sorted(notes, key=lambda n: n["filepath"])


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
    path = Path(path)
    if path.name.startswith("."):
        log.info("Skipping hidden file: %s", path)
        return None
    if not path.is_file() or path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        log.info("Skipping unsupported or missing file: %s", path)
        return None

    try:
        resolved = path.resolve()
        mtime = datetime.fromtimestamp(resolved.stat().st_mtime)
        raw_content = _read_text_by_extension(resolved)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read file %s: %s", path, exc)
        return None

    content = raw_content.strip()
    if not content:
        log.info("Skipping empty or unreadable file: %s", resolved)
        return None

    title = _extract_title(content, resolved.stem)
    client = extract.guess_client(content) or _client_from_path(resolved)
    tags = extract.extract_tags(content)
    event_time = extract.extract_event_time(content, reference=mtime)

    note = NoteItem(
        id=make_note_id(resolved),
        title=title,
        content=content,
        filepath=str(resolved),
        timestamp=mtime,
        client=client,
        tags=tags,
        event_time=event_time,
    )
    return note_to_dict(note)