# =============================================================================
# module_rag/guardrails.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : retrieved ContextItems (from Person 2 via pipeline.py) and the raw
#            LLM answer (from llm_client.chat via engine.py).
# Hands off: SYSTEM_PROMPT and pass/fail checks back to engine.py, which turns
#            any failure into the standard fallback AnswerDict for Person 4.
# STATUS   : minimal REAL checks (the tests rely on them); harden as needed.
# =============================================================================
"""Prompt and checks that stop the assistant from inventing answers."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import config
from contracts import FALLBACK_MESSAGE, ContextItem

SYSTEM_PROMPT = f"""You are a private, offline assistant for a busy manager. You answer questions about the manager's own notes.

Rules:
1. Answer ONLY from the notes inside the CONTEXT block. Each note starts with a line like: SOURCE [filename]
2. Cite the source filename in square brackets after every fact, e.g. [call_prep.txt]. Use the exact filename shown after SOURCE.
3. If the CONTEXT does not contain the answer, reply with exactly this sentence and nothing else: {FALLBACK_MESSAGE}
4. Never use outside knowledge. Never guess names, dates, times, places or numbers.
5. If the question is ambiguous (for example two clients match), do not guess: ask which one the user means, and cite the notes that mention each option.
6. Keep answers short: one to three sentences unless the user asks for detail.
"""

_BRACKET_RE = re.compile(r"\[([^\[\]\n]+)\]")


def extract_cited_filenames(answer: str) -> list[str]:
    """Return the filenames cited in square brackets, in order, de-duplicated.

    Only bracket contents ending in a supported extension count as citations
    ("[call_prep.txt]", "[a.md, b.txt]"); "[1]" or "[note]" are ignored.
    Paths inside brackets are reduced to their filename.
    """
    # TODO(person 3): handle quotes/odd spacing the model produces in practice.
    names: list[str] = []
    for group in _BRACKET_RE.findall(answer or ""):
        for part in re.split(r"[,;]", group):
            part = part.strip().strip("'\"`")
            name = PurePosixPath(part.replace("\\", "/")).name
            if PurePosixPath(name).suffix.lower() in config.SUPPORTED_EXTENSIONS and name not in names:
                names.append(name)
    return names


def cited_sources(answer: str, context: list[ContextItem]) -> list[str]:
    """Map the filenames cited in ``answer`` to context filepaths (context order)."""
    # TODO(person 3): if two context notes share a filename, both are returned.
    cited = {n.lower() for n in extract_cited_filenames(answer)}
    sources: list[str] = []
    for item in context:
        if _filename(item["filepath"]).lower() in cited and item["filepath"] not in sources:
            sources.append(item["filepath"])
    return sources


def has_enough_context(context: list[ContextItem]) -> bool:
    """Decide whether retrieval found anything worth sending to the LLM.

    Inputs: retrieved ContextItems (scores 0..1).
    Output: False if ``context`` is empty or the best score is below
        config.MIN_SIMILARITY; True otherwise.
    Edge cases: missing/None score counts as 0.
    """
    # TODO(person 3): tune config.MIN_SIMILARITY on the sample notes.
    if not context:
        return False
    best = max(float(item.get("score") or 0.0) for item in context)
    return best >= config.MIN_SIMILARITY


def validate_citations(answer: str, context: list[ContextItem]) -> bool:
    """Check that the answer cites real sources from the context.

    Inputs: the LLM answer and the context it was given.
    Output: True only if at least one filename is cited AND every cited
        filename belongs to a context item (case-insensitive, by filename).
    Edge cases: empty answer -> False; empty context -> False.
    """
    # TODO(person 3): two notes can share a filename in different folders;
    # decide whether to cite relative paths instead.
    cited = extract_cited_filenames(answer)
    if not cited or not context:
        return False
    available = {_filename(item["filepath"]).lower() for item in context}
    return all(name.lower() in available for name in cited)


def _filename(filepath: str) -> str:
    return PurePosixPath(filepath.replace("\\", "/")).name
