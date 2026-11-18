"""Launch the chat UI with the team's local RAG pipeline."""

import config

from datetime import datetime, timedelta
from functools import partial
import re
from typing import Optional, Sequence

import streamlit as st

from contracts import fallback_answer
from module_ui.app import add_messages, main
from module_ui.chat_service import ChatReply, Message, SourceReference
from module_ingest.parser import scan_and_parse
from module_rag import llm_client, pipeline
from module_storage.store import NotesStore
from module_ui import notifier, ui_app

SOCIAL_REPLIES = {
    **dict.fromkeys(
        ("hi", "hello", "hey", "hi there", "hello there", "good morning", "good evening"),
        "Hi! What would you like to find in your notes?",
    ),
    **dict.fromkeys(("thanks", "thank you"), "You're welcome!"),
    **dict.fromkeys(
        ("how are you", "how r u", "how are u", "how r you", "how's it going"),
        "I'm doing well, thanks for asking! What can I help you find?",
    ),
    "what can you do": (
        "I can answer questions about your local notes and show the sources I used. "
        "Try asking about a meeting, a client, or a task."
    ),
}

UPCOMING_EVENTS_QUERY = re.compile(
    r"(?:(?:do (?:you|u) know (?:about|of)|are there|show me|list|any|what are|"
    r"what|which) )?(?:the |any )?"
    r"(?:upcoming (?:events?|meetings?|appointments?)|"
    r"(?:events?|meetings?|appointments?) (?:coming up|"
    r"(?:in (?:the )?)?(?:incoming|coming|next|upcoming) (?:few )?months))"
)


def normalize_query(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip("!?.").strip()


def upcoming_events(notes: list[dict]) -> ChatReply:
    now = datetime.now().astimezone()
    cutoff = now + timedelta(days=90)
    events = []
    for note in notes:
        event_time = note.get("event_time")
        if isinstance(event_time, datetime):
            local_time = event_time.astimezone()
            if now <= local_time <= cutoff:
                events.append((local_time, note))
    if not events:
        return ChatReply(
            "I found no dated events in your notes for the next 90 days. "
            "If you know the event name, try asking about it directly."
        )

    events.sort(key=lambda event: event[0])
    lines = [f"I found {len(events)} dated event(s) in your notes for the next 90 days."]
    sources = []
    for number, (event_time, note) in enumerate(events[:10], start=1):
        lines.append(f"- {event_time:%d %b %Y, %H:%M} · {note['title']} [{number}]")
        sources.append(SourceReference(note["title"], note["filepath"], note["content"]))
    if len(events) > len(sources):
        lines.append("\nShowing the next 10 events.")
    return ChatReply("\n".join(lines), tuple(sources), grounded=True)


class RagChatService:
    def __init__(self, store: NotesStore) -> None:
        self.store = store

    def reply(
        self, messages: Sequence[Message]
    ) -> ChatReply:
        if not messages or not messages[-1].text.strip():
            return ChatReply.from_answer(fallback_answer(), ())
        query = messages[-1].text
        normalized = normalize_query(query)
        reply = SOCIAL_REPLIES.get(normalized)
        if reply:
            return ChatReply(reply)
        if UPCOMING_EVENTS_QUERY.fullmatch(normalized):
            return upcoming_events(self.store.all_notes())

        history = [
            {"role": message.role, "content": message.text}
            for message in messages[:-1]
            if (message.role == "user" and normalize_query(message.text) not in SOCIAL_REPLIES)
            or (message.role == "assistant" and message.grounded is not None)
        ]
        answer = pipeline.ask(
            query=query,
            chat_history=history,
            store=self.store,
            pinned_note_ids=list(st.session_state.get("pinned_note_ids", [])),
        )
        notes = self.store.all_notes() if answer["sources"] else ()
        return ChatReply.from_answer(answer, notes)


@st.cache_resource(show_spinner="Indexing local notes…")
def load_store() -> tuple[NotesStore, list[dict]]:
    notes = scan_and_parse(config.NOTE_DIR)
    store = NotesStore()
    store.add(notes)
    notifier.start_scheduler(store.all_notes)
    return store, notes


def add_reminder_to_chat(reminder: dict) -> None:
    content = f"Reminder: {reminder['title']} [{reminder['filename']}]\n\n{reminder['content']}"
    source = SourceReference(reminder["title"], reminder["filepath"], reminder["content"])
    add_messages(Message("assistant", content, (source,), grounded=True))
    ui_app.pin_note(reminder["note_id"])


def render(store: NotesStore, notes: list[dict]) -> None:
    ui_app.init_session_state()
    status = "Local RAG pipeline" if llm_client.is_available() else "Local model unavailable"
    main(
        RagChatService(store),
        status=status,
        render_notes=partial(ui_app.render_sidebar, store, notes),
        render_reminders=partial(ui_app.reminder_panel, add_reminder_to_chat),
    )


def run() -> None:
    render(*load_store())
