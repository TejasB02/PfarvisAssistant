"""Launch the chat UI with the team's local RAG pipeline."""

import config

from functools import partial
from typing import Sequence

import streamlit as st

from contracts import fallback_answer
from module_ui.app import add_messages, main
from module_ui.chat_service import ChatReply, Message, SourceReference
import pipeline as ingest_pipeline
from module_rag import llm_client, pipeline
from module_storage.store import NotesStore
from module_ui import notifier, ui_app

class RagChatService:
    def __init__(self, store: NotesStore) -> None:
        self.store = store

    def reply(
        self, messages: Sequence[Message]
    ) -> ChatReply:
        if not messages or not messages[-1].text.strip():
            return ChatReply.from_answer(fallback_answer(), ())
        query = messages[-1].text
        history = [
            {"role": message.role, "content": message.text}
            for message in messages[:-1]
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
    # Initial full index, then a background watcher keeps the vector DB in sync
    # with every create / edit / delete under config.NOTE_DIR.
    store = ingest_pipeline.run_ingestion_pipeline(config.NOTE_DIR, store=NotesStore(), live_sync=True)
    notifier.start_scheduler(store.all_notes)
    return store, store.all_notes()


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
        render_reminders=partial(ui_app.reminder_panel, add_reminder_to_chat),
    )


def run() -> None:
    store, _ = load_store()
    # Re-read on every rerun so the sidebar reflects live-synced changes.
    notes = sorted(store.all_notes(), key=lambda n: n["filepath"])
    render(store, notes)
