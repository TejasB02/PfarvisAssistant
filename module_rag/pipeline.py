# =============================================================================
# module_rag/pipeline.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : query, chat history and pinned note ids from Person 4 (UI), plus
#            the NotesStore built by app.py (Person 2's class).
# Hands off: AnswerDict -> Person 4. ask() is the ONLY function the UI calls
#            for answers.
# Flow     : rewrite_query -> store.hybrid_search + notes whose event_time is
#            in the time span asked about (dates.py) -> add pinned notes ->
#            engine.answer_user_query
# =============================================================================
"""End-to-end question answering used by the UI."""

from __future__ import annotations

import logging
from datetime import datetime

import config
from contracts import AnswerDict, ContextItem, fallback_answer
from module_rag import dates, engine, llm_client
from module_storage.store import NotesStore

log = logging.getLogger(__name__)

MAX_DATED_NOTES = 5           # notes added by the date check (see _notes_in_time_span)
REWRITE_HISTORY_MESSAGES = 6
REWRITE_CHARS_PER_MESSAGE = 600

REWRITE_PROMPT = """You rewrite the user's latest message as ONE standalone question that can be used to search their notes.
Use only names and details that appear in the conversation. Do not answer the question. Do not add anything new.
If the latest message is already a standalone question, return it unchanged.
Reply with the question only, on a single line.

Example: after a conversation about the 4pm call today, "Which client?" becomes:
Which client is the 4pm call today with?"""


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
        - LLM error, empty reply, or a reply longer than 3x the query + 200
          characters -> return ``query`` unchanged.
    """
    if not chat_history or not query.strip():
        return query

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:REWRITE_CHARS_PER_MESSAGE]}"
        for m in chat_history[-REWRITE_HISTORY_MESSAGES:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {query}\n\nStandalone question:"},
    ]
    try:
        reply = llm_client.chat(messages, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - a failed rewrite must not break answering
        log.warning("rewrite_query: LLM call failed (%s); using the original query", exc)
        return query

    rewritten = _clean_rewrite(reply)
    if not rewritten or len(rewritten) > 3 * len(query) + 200:
        log.info("rewrite_query: unusable rewrite %r; keeping %r", (reply or "")[:200], query)
        return query
    log.info("rewrite_query: %r -> %r", query, rewritten)
    return rewritten


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
          filepath, with score 0.0 and via="graph": they reach the prompt but
          cannot pass has_enough_context on their own.
    """
    if not query or not query.strip():
        return fallback_answer()

    standalone = rewrite_query(query, chat_history)
    searched = list(store.hybrid_search(standalone, k=config.RETRIEVAL_TOP_K, hops=config.GRAPH_HOPS))
    dated = _notes_in_time_span(standalone, store)
    dated_paths = {item["filepath"] for item in dated}
    context: list[ContextItem] = dated + [c for c in searched if c["filepath"] not in dated_paths]
    if dated:
        log.info("ask: %d note(s) with an event in the time span asked about", len(dated))
    best = max((c["score"] for c in context if c["via"] == "semantic"), default=0.0)
    log.info("ask: best semantic score %.3f for %r (MIN_SIMILARITY=%.2f)",
             best, standalone, config.MIN_SIMILARITY)

    seen = {item["filepath"] for item in context}
    for note_id in pinned_note_ids or []:
        note = store.get_note(note_id)
        if note is None or note["filepath"] in seen:
            continue
        # pinned notes: score 0.0 so they never open the gate
        context.append(ContextItem(
            filepath=note["filepath"], title=note["title"], content=note["content"],
            score=0.0, via="graph",
        ))
        seen.add(note["filepath"])

    return engine.answer_user_query(query, context, chat_history)


def _notes_in_time_span(question: str, store: NotesStore) -> list[ContextItem]:
    """Notes whose event_time falls in the span the question names ("in December",
    "this week"...), soonest first, at most MAX_DATED_NOTES.

    They count as exact semantic hits (score 1.0, via="semantic"): the date match
    is on Person 1's parsed event_time, which embeddings cannot see. No span in
    the question, or a store that can't list its notes -> [].
    """
    matches = dates.event_filter(question)
    if matches is None:
        return []
    try:
        notes = list(store.all_notes())
    except Exception as exc:  # noqa: BLE001 - the date check is an extra, never a blocker
        log.warning("ask: could not list notes for the date check (%s)", exc)
        return []
    dated = sorted(
        (n for n in notes if isinstance(n.get("event_time"), datetime) and matches(n["event_time"])),
        key=lambda n: n["event_time"],
    )[:MAX_DATED_NOTES]
    return [ContextItem(filepath=n["filepath"], title=n["title"], content=n["content"],
                        score=1.0, via="semantic") for n in dated]


def _clean_rewrite(reply: str) -> str:
    """First non-empty line, without labels or surrounding quotes."""
    for line in (reply or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for label in ("standalone question:", "question:", "rewritten:"):
            if line.lower().startswith(label):
                line = line[len(label):].strip()
        return line.strip("\"'` ")
    return ""
