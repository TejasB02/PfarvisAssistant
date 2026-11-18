# =============================================================================
# module_rag/pipeline.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : query, chat history and pinned note ids from Person 4 (UI), plus
#            the NotesStore built by app.py (Person 2's class).
# Hands off: AnswerDict -> Person 4. ask() is the ONLY function the UI calls
#            for answers.
# Flow     : rewrite_query -> store.hybrid_search -> add pinned notes ->
#            engine.answer_user_query
# STATUS   : flow is REAL glue; rewrite_query is a pass-through stub.
# =============================================================================
"""End-to-end question answering used by the UI."""

from __future__ import annotations

import logging

import config
from contracts import AnswerDict, ContextItem, fallback_answer
from module_rag import engine
from module_storage.store import NotesStore

log = logging.getLogger(__name__)


def rewrite_query(query: str, chat_history: list[dict[str, str]]) -> str:
    """Turn a follow-up into a standalone question for retrieval.

    Example: history "When is my call with the clients?" + "Which client?"
    -> "Which client is the 4pm call today with?"
    Inputs:
        query: latest user message.
        chat_history: previous turns [{"role", "content"}].
    Output:
        Standalone question string, used ONLY for retrieval (the engine still
        sees the original query plus history).
    Edge cases:
        - Empty history -> return ``query`` unchanged (no LLM call).
        - LLM unavailable or returns junk/empty -> return ``query`` unchanged.
        - Must never add facts that are not in the history.
    """
    # TODO(person 3): use llm_client.chat with a short rewrite prompt at
    # temperature 0 when chat_history is not empty.
    return query


def ask(
    query: str,
    chat_history: list[dict[str, str]],
    store: NotesStore,
    pinned_note_ids: list[str] | None = None,
) -> AnswerDict:
    """Answer a user question from the notes. The ONLY function the UI calls.

    Inputs:
        query: latest user message.
        chat_history: previous turns [{"role", "content"}], oldest first,
            excluding ``query`` itself.
        store: the app's NotesStore.
        pinned_note_ids: notes the user pinned (e.g. from a reminder), always
            added to the context so follow-ups stay about the right note.
    Output: AnswerDict (see contracts.py).
    Edge cases:
        - Blank query -> fallback without retrieval.
        - Pinned ids that no longer exist are ignored.
        - Pinned notes are appended after search results, de-duplicated by
          filepath, with score 1.0 (the user chose them) and via="graph".
    """
    # TODO(person 3): keep this flow; tune k/hops via config.
    if not query or not query.strip():
        return fallback_answer()

    standalone = rewrite_query(query, chat_history)
    context: list[ContextItem] = list(
        store.hybrid_search(standalone, k=config.RETRIEVAL_TOP_K, hops=config.GRAPH_HOPS)
    )
    seen = {item["filepath"] for item in context}
    for note_id in pinned_note_ids or []:
        note = store.get_note(note_id)
        if note is None or note["filepath"] in seen:
            continue
        context.append(ContextItem(
            filepath=note["filepath"], title=note["title"], content=note["content"],
            score=1.0, via="graph",
        ))
        seen.add(note["filepath"])

    return engine.answer_user_query(query, context, chat_history)
