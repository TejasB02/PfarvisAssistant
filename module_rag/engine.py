# =============================================================================
# module_rag/engine.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : query + list[ContextItem] (from Person 2's NotesStore, via
#            pipeline.ask) + chat history (from Person 4's UI, via pipeline.ask).
# Hands off: AnswerDict -> pipeline.ask -> Person 4 (UI).
# Rules    : no context -> fallback WITHOUT calling the LLM; bad or missing
#            citations -> fallback. Fallback = contracts.fallback_answer().
# STATUS   : guard flow is REAL; the LLM behind llm_client.chat is a mock.
# =============================================================================
"""Grounded answering: prompt -> local LLM -> citation check."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from contracts import FALLBACK_MESSAGE, AnswerDict, ContextItem, fallback_answer
from module_rag import guardrails, llm_client

log = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 6      # last N chat turns sent to the model
MAX_CHARS_PER_NOTE = 2000     # keep the prompt small for local models


def format_context(context: list[ContextItem]) -> str:
    """Render the CONTEXT block. Each note starts with "SOURCE [filename]"."""
    # TODO(person 3): smarter truncation (keep the most relevant chunk).
    blocks = []
    for item in context:
        path = PurePosixPath(item["filepath"].replace("\\", "/"))
        blocks.append(
            f"SOURCE [{path.name}]\n"
            f"Title: {item['title']}\n"
            f"Folder: {path.parent.name}\n"
            f"{item['content'].strip()[:MAX_CHARS_PER_NOTE]}"
        )
    return "\n\n---\n\n".join(blocks)


def build_messages(
    query: str,
    retrieved_context: list[ContextItem],
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the Ollama message list: system prompt, recent history, then a
    final user turn holding the CONTEXT block and the QUESTION."""
    # TODO(person 3): tune how much history to keep.
    messages = [{"role": "system", "content": guardrails.SYSTEM_PROMPT}]
    for msg in chat_history[-MAX_HISTORY_MESSAGES:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": (
            f"CONTEXT:\n{format_context(retrieved_context)}\nEND OF CONTEXT\n\n"
            f"QUESTION: {query}"
        ),
    })
    return messages


def answer_user_query(
    query: str,
    retrieved_context: list[ContextItem],
    chat_history: list[dict[str, str]],
) -> AnswerDict:
    """Answer ``query`` strictly from ``retrieved_context``. THE Person 3 deliverable.

    Inputs:
        query: the user's question (may be a follow-up like "Which client?").
        retrieved_context: ContextItems from NotesStore.hybrid_search (+ pinned).
        chat_history: previous turns as [{"role", "content"}] (strings only).
    Output:
        AnswerDict. ``grounded`` is True only when the answer passed
        validate_citations; ``sources`` are the cited context filepaths.
    Edge cases (all return contracts.fallback_answer()):
        - has_enough_context False -> fallback, LLM is NOT called.
        - LLM raises (server down, timeout) -> logged, fallback.
        - LLM replies with FALLBACK_MESSAGE -> fallback.
        - validate_citations False (no citation, or a file not in context) -> fallback.
    """
    # TODO(person 3): the guard flow below is the contract; improve the prompt,
    # history handling and clarifying-question behaviour around it.
    if not guardrails.has_enough_context(retrieved_context):
        return fallback_answer()

    messages = build_messages(query, retrieved_context, chat_history)
    try:
        raw = llm_client.chat(messages, temperature=0.0)
    except Exception:  # noqa: BLE001 - any LLM failure must degrade safely
        log.exception("Local LLM call failed")
        return fallback_answer()

    answer = (raw or "").strip()
    if not answer or FALLBACK_MESSAGE.lower() in answer.replace("’", "'").lower():
        return fallback_answer()
    if not guardrails.validate_citations(answer, retrieved_context):
        log.info("Answer rejected by citation check: %r", answer[:200])
        return fallback_answer()

    return {
        "answer": answer,
        "sources": guardrails.cited_sources(answer, retrieved_context),
        "grounded": True,
    }
