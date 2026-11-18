# =============================================================================
# scripts/search_notes.py - shared debugging tool (Person 2 / Person 3)
# Receives : a question on the command line.
# Hands off: a printed list of what NotesStore.hybrid_search returns for it:
#            score, semantic/graph, filename, title. Fully local.
# Run      : python scripts/search_notes.py "Do I have any meetings in December?"
#            python scripts/search_notes.py --k 10 --hops 0 "parking policy"
#            python scripts/search_notes.py --list        (every indexed note)
# =============================================================================
"""Search the notes database the same way the app does, and print the hits."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402  - first project import: offline env vars

from module_ingest.parser import scan_and_parse  # noqa: E402
from module_storage.store import NotesStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="what to search for")
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_TOP_K, help="semantic hits (default %(default)s)")
    parser.add_argument("--hops", type=int, default=config.GRAPH_HOPS, help="graph hops (default %(default)s)")
    parser.add_argument("--list", action="store_true", help="list every indexed note instead of searching")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.ERROR)

    store = NotesStore()
    store.add(scan_and_parse(config.NOTE_DIR))
    notes = store.all_notes()
    print(f"{len(notes)} notes indexed from {config.NOTE_DIR}\n")

    if args.list or not args.question:
        for note in sorted(notes, key=lambda n: n["filepath"]):
            when = f"  event {note['event_time']:%Y-%m-%d %H:%M}" if note.get("event_time") else ""
            print(f"- {Path(note['filepath']).name:32} {note['title'][:40]:40} "
                  f"client={note.get('client') or '-'}  tags={','.join(note.get('tags') or []) or '-'}{when}")
        return 0

    question = " ".join(args.question)
    hits = store.hybrid_search(question, k=args.k, hops=args.hops)
    print(f"Q: {question}   (k={args.k}, hops={args.hops}, gate MIN_SIMILARITY={config.MIN_SIMILARITY})")
    if not hits:
        print("  no results")
    for hit in hits:
        passes = "  <- opens the gate" if hit["via"] == "semantic" and hit["score"] >= config.MIN_SIMILARITY else ""
        print(f"  {hit['score']:.3f}  {hit['via']:8}  {Path(hit['filepath']).name:32} {hit['title'][:40]}{passes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
