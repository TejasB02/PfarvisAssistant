# =============================================================================
# module_rag/guardrails.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : retrieved ContextItems (from Person 2 via pipeline.py) and the raw
#            LLM answer (from llm_client.chat via engine.py).
# Hands off: SYSTEM_PROMPT and pass/fail checks back to engine.py, which turns
#            any failure into the standard fallback AnswerDict for Person 4.
# =============================================================================
"""Prompt and checks that stop the assistant from inventing answers."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Optional

import config
from contracts import FALLBACK_MESSAGE, ContextItem

SYSTEM_PROMPT = f"""You answer questions about the user's own notes. You know NOTHING except the notes in the CONTEXT block.

Rules:
1. Use only facts written in the CONTEXT notes. No outside knowledge. No guessing.
2. After each fact, cite the note it came from: its filename in square brackets, exactly as written after SOURCE, e.g. [team_lunch.md].
3. If the CONTEXT does not contain the answer, reply with exactly this sentence and nothing else:
{FALLBACK_MESSAGE}
4. If the question could refer to more than one thing in the notes (for example "the client" when the notes mention two clients), do NOT answer for each one. Reply only with a short question asking which one the user means, list every option, and cite every note you mention. End with a question mark.
5. Earlier chat messages only help you understand follow-up questions. Facts must still come from the CONTEXT.
6. Answer in one to three short sentences.

Example of a correct answer:
The team lunch is on Friday at 1pm [team_lunch.md].

Example when the notes do not contain the answer:
{FALLBACK_MESSAGE}

Example when two notes could match:
Which supplier do you mean: Pellow Paper [pellow.md] or Brisk Inks [brisk_inks.md]?
"""

# [call_prep.txt]  or  (call_prep.txt)
_CITATION_RE = re.compile(r"\[([^\[\]\n]+)\]|\(([^()\n]+)\)")
_PREFIX_RE = re.compile(r"^(?:sources?|files?|notes?|from|see)\s*[:=-]\s*", re.IGNORECASE)
_STRIP_CHARS = " \t'\"`*_"


def extract_cited_filenames(answer: str, context: Optional[list[ContextItem]] = None) -> list[str]:
    """Return the filenames cited in ``answer``, in order, de-duplicated.

    Accepted citation shapes (all seen from local models):
        [call_prep.txt]   (call_prep.txt)   [call_prep]   [meetings/2026-11/call_prep.txt]
        [a.md, b.txt]     [Source: call_prep.txt]
    Rules:
        - A token WITH a supported extension is always a citation. If it matches
          a context filename (case-insensitive) the context spelling is returned,
          otherwise the token's own filename (so validate_citations rejects it).
        - A token WITHOUT an extension counts only if it equals the stem of a
          context filename ("[call_prep]"); "[1]", "(virtual)" etc. are ignored.
        - Paths (/ or \\) are reduced to their filename.
    """
    by_name, by_stem = _context_index(context or [])
    names: list[str] = []
    for token in _citation_tokens(answer):
        name = PurePosixPath(token.replace("\\", "/")).name.strip(_STRIP_CHARS)
        if not name:
            continue
        lower = name.lower()
        has_extension = PurePosixPath(lower).suffix in config.SUPPORTED_EXTENSIONS
        if lower in by_name:
            resolved = by_name[lower]
        elif not has_extension and lower in by_stem:
            resolved = by_stem[lower]
        elif has_extension:
            resolved = name  # a file that is not in the context
        else:
            continue  # not a citation
        if resolved not in names:
            names.append(resolved)
    return names


def cited_sources(answer: str, context: list[ContextItem]) -> list[str]:
    """Map the filenames cited in ``answer`` to context filepaths (context order).

    If two context notes share a filename, both filepaths are returned.
    """
    cited = {n.lower() for n in extract_cited_filenames(answer, context)}
    sources: list[str] = []
    for item in context:
        if _filename(item["filepath"]).lower() in cited and item["filepath"] not in sources:
            sources.append(item["filepath"])
    return sources


def has_enough_context(context: list[ContextItem]) -> bool:
    """Decide whether retrieval found anything worth sending to the LLM.

    Inputs: retrieved ContextItems (scores 0..1).
    Output: True only if at least one item with via == "semantic" scores at
        least config.MIN_SIMILARITY. Graph neighbours and pinned notes still go
        into the prompt but cannot open the gate on their own.
    Edge cases: empty context, or no semantic items -> False; missing/None
        score counts as 0.
    """
    # pinned notes: score 0.0 so they never open the gate
    semantic = [float(item.get("score") or 0.0) for item in context if item.get("via") == "semantic"]
    return bool(semantic) and max(semantic) >= config.MIN_SIMILARITY


def validate_citations(answer: str, context: list[ContextItem]) -> bool:
    """Check that the answer cites real sources from the context.

    Inputs: the LLM answer and the context it was given.
    Output: True only if at least one note is cited AND every cited filename
        belongs to a context item (case-insensitive; extension optional, see
        extract_cited_filenames).
    Edge cases: empty answer -> False; empty context -> False.
    """
    if not context:
        return False
    cited = extract_cited_filenames(answer, context)
    if not cited:
        return False
    available = {_filename(item["filepath"]).lower() for item in context}
    return all(name.lower() in available for name in cited)


# --- helpers --------------------------------------------------------------------

def _citation_tokens(answer: str):
    for match in _CITATION_RE.finditer(answer or ""):
        group = match.group(1) if match.group(1) is not None else match.group(2)
        for part in re.split(r"[,;]", group):
            token = _PREFIX_RE.sub("", part.strip(_STRIP_CHARS)).strip(_STRIP_CHARS)
            if token:
                yield token


def _context_index(context: list[ContextItem]) -> tuple[dict[str, str], dict[str, str]]:
    by_name: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for item in context:
        name = _filename(item["filepath"])
        by_name.setdefault(name.lower(), name)
        by_stem.setdefault(PurePosixPath(name).stem.lower(), name)
    return by_name, by_stem


def _filename(filepath: str) -> str:
    return PurePosixPath(filepath.replace("\\", "/")).name
