# =============================================================================
# module_ui/ui_app.py - owner: Person 4 (Frontend, State & Notifications)
# Receives : NotesStore (Person 2) and note dicts (Person 1) from app.py;
#            AnswerDicts from Person 3 via pipeline.ask; reminder dicts from
#            notifier.pop_pending_reminders.
# Hands off: user questions + chat history + pinned note ids -> pipeline.ask
#            (Person 3); the rendered page -> the manager.
# Session  : st.session_state["messages"]         chat history (+ sources)
#            st.session_state["pinned_note_ids"]  passed to pipeline.ask
#            st.session_state["active_reminders"] reminder cards on screen
# STATUS   : basic working layout so the app runs end to end; polish it.
# =============================================================================
"""Streamlit page: chat, notes sidebar, reminder cards."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

import config
from module_rag import llm_client, pipeline
from module_storage.store import NotesStore
from module_ui import notifier

REMINDER_POLL_SECONDS = 5
NO_CLIENT_LABEL = "No client"


def main(store: NotesStore, notes: list[dict]) -> None:
    """Render the whole page. Called by app.py on every Streamlit rerun.

    Inputs:
        store: the shared NotesStore (cached by app.py).
        notes: note dicts from scan_and_parse (cached by app.py).
    Output: None (renders UI).
    Edge cases: no notes -> sidebar says so, chat still works (fallback answers).
    """
    # TODO(person 4): polish layout, empty states, auto-pin top source of a
    # grounded answer, streaming answers.
    init_session_state()
    render_sidebar(store, notes)
    st.title("Local Notes Assistant")
    st.caption("Answers come only from your notes. Everything stays on this machine.")
    reminder_panel()
    render_chat(store)


def init_session_state() -> None:
    """Create the session keys this page relies on (idempotent)."""
    # TODO(person 4): keep keys in sync with the header comment.
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pinned_note_ids", [])
    st.session_state.setdefault("active_reminders", [])


# --- Chat ----------------------------------------------------------------------

def render_chat(store: NotesStore) -> None:
    """Show history, read st.chat_input, call pipeline.ask, store the reply."""
    # TODO(person 4): show a "not found" style for grounded=False answers.
    for message in st.session_state["messages"]:
        render_message(message)

    prompt = st.chat_input("Ask about your notes, e.g. 'When is my client call?'")
    if not prompt:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
    user_msg = {"role": "user", "content": prompt, "sources": []}
    st.session_state["messages"].append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            result = pipeline.ask(
                prompt, history, store,
                pinned_note_ids=list(st.session_state["pinned_note_ids"]),
            )
        st.markdown(result["answer"])
        render_sources(result["sources"])
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": list(result["sources"]),
        "grounded": result["grounded"],
    })


def render_message(message: dict) -> None:
    """Render one chat message; assistant messages get a sources expander."""
    # TODO(person 4): style reminder messages differently from answers.
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])


def render_sources(sources: list[str]) -> None:
    """Small expander listing the source filepaths of an answer."""
    # TODO(person 4): show paths relative to NOTE_DIR (display_folder) for readability.
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.caption(f"`{source}`")


# --- Reminders -----------------------------------------------------------------

@st.fragment(run_every=REMINDER_POLL_SECONDS)
def reminder_panel(on_add: Callable[[dict], None] | None = None) -> None:
    """Poll the notifier queue, toast new reminders, show cards with "Add to chat".

    Runs as a fragment so only this block reruns every few seconds.
    """
    # TODO(person 4): style the cards; consider a snooze button.
    for reminder in notifier.pop_pending_reminders():
        st.toast(reminder["message"], icon="⏰")
        st.session_state["active_reminders"].append(reminder)

    for reminder in list(st.session_state["active_reminders"]):
        with st.container(border=True):
            st.markdown(f"**⏰ {reminder['message']}**")
            st.caption(reminder["filename"])
            add_col, dismiss_col = st.columns(2)
            if add_col.button("Add to chat", key=f"rem_add_{reminder['note_id']}"):
                if on_add is None:
                    inject_reminder(reminder)
                else:
                    on_add(reminder)
                    _drop_reminder(reminder["note_id"])
                st.rerun()  # full rerun so the chat shows the new message
            if dismiss_col.button("Dismiss", key=f"rem_dismiss_{reminder['note_id']}"):
                _drop_reminder(reminder["note_id"])
                st.rerun()


def inject_reminder(reminder: dict) -> None:
    """Append "Reminder: <title> [<filename>]" + note content as an assistant
    message, pin the note so follow-ups are about it, and remove the card."""
    # TODO(person 4): keep the message format; Person 3's prompt relies on it.
    content = f"Reminder: {reminder['title']} [{reminder['filename']}]\n\n{reminder['content']}"
    st.session_state["messages"].append({
        "role": "assistant", "content": content,
        "sources": [reminder["filepath"]], "grounded": True,
    })
    pin_note(reminder["note_id"])
    _drop_reminder(reminder["note_id"])


def _drop_reminder(note_id: str) -> None:
    st.session_state["active_reminders"] = [
        r for r in st.session_state["active_reminders"] if r["note_id"] != note_id
    ]


# --- Sidebar -------------------------------------------------------------------

def render_sidebar(store: NotesStore, notes: list[dict]) -> None:
    """Notes grouped by client, a text filter, pinned notes and LLM status."""
    # TODO(person 4): show folder tree, counts, "rescan" button.
    with st.sidebar:
        st.header("Your notes")
        st.caption(f"{len(notes)} notes from `{config.NOTE_DIR}`")
        if llm_client.is_available():
            st.caption(f"Local model: {config.OLLAMA_MODEL} (ready)")
        else:
            st.warning(f"Local model {config.OLLAMA_MODEL} is not running. Start Ollama.")

        pinned = st.session_state["pinned_note_ids"]
        if pinned:
            st.subheader("Pinned for follow-ups")
            for note_id in list(pinned):
                note = store.get_note(note_id)
                label = note["title"] if note else note_id[:8]
                st.button(f"✕ {label}", key=f"unpin_{note_id}", on_click=unpin_note, args=(note_id,))

        query = st.text_input("Filter notes", key="note_filter", placeholder="client, tag or word")
        groups = group_notes_by_client(filter_notes(notes, query))
        if not groups:
            st.info("No notes match." if notes else "No notes found yet.")
        for client, group in groups.items():
            st.subheader(f"{client} ({len(group)})")
            for note in group:
                render_note_card(note)


def render_note_card(note: dict) -> None:
    """One expander: title, folder, event_time, tags and a pin toggle."""
    # TODO(person 4): show the client and a short content preview.
    with st.expander(note["title"]):
        st.caption(f"Folder: {display_folder(note['filepath'])}")
        if note.get("event_time"):
            st.caption(f"When: {format_event_time(note['event_time'])}")
        if note.get("tags"):
            st.caption("Tags: " + " ".join(f"#{t}" for t in note["tags"]))
        is_pinned = note["id"] in st.session_state["pinned_note_ids"]
        st.button(
            "Unpin" if is_pinned else "Pin to chat",
            key=f"pin_{note['id']}",
            on_click=unpin_note if is_pinned else pin_note,
            args=(note["id"],),
        )


def pin_note(note_id: str) -> None:
    """Add a note id to st.session_state["pinned_note_ids"] (no duplicates)."""
    # TODO(person 4): cap the number of pinned notes to keep the prompt small.
    if note_id not in st.session_state["pinned_note_ids"]:
        st.session_state["pinned_note_ids"].append(note_id)


def unpin_note(note_id: str) -> None:
    """Remove a note id from st.session_state["pinned_note_ids"]."""
    # TODO(person 4): nothing else to do unless pins gain extra state.
    st.session_state["pinned_note_ids"] = [
        n for n in st.session_state["pinned_note_ids"] if n != note_id
    ]


# --- Pure helpers (easy to unit test) ------------------------------------------

def filter_notes(notes: list[dict], text: str) -> list[dict]:
    """Case-insensitive match on title, content, client, tags and filepath."""
    # TODO(person 4): support "#tag" and "client:" filter syntax.
    needle = (text or "").strip().lower()
    if not needle:
        return list(notes)
    def haystack(n: dict) -> str:
        return " ".join([n["title"], n["content"], n.get("client") or "",
                         " ".join(n.get("tags") or []), n["filepath"]]).lower()
    return [n for n in notes if needle in haystack(n)]


def group_notes_by_client(notes: list[dict]) -> dict[str, list[dict]]:
    """{client: notes}, clients alphabetical, "No client" last, notes by title."""
    # TODO(person 4): consider sorting notes by event_time within a client.
    groups: dict[str, list[dict]] = {}
    for note in notes:
        groups.setdefault(note.get("client") or NO_CLIENT_LABEL, []).append(note)
    ordered = sorted(groups, key=lambda c: (c == NO_CLIENT_LABEL, c.lower()))
    return {c: sorted(groups[c], key=lambda n: n["title"].lower()) for c in ordered}


def display_folder(filepath: str) -> str:
    """Folder relative to NOTE_DIR when possible, else the absolute folder."""
    # TODO(person 4): shorten very deep paths for the sidebar.
    folder = Path(filepath).parent
    try:
        relative = folder.relative_to(Path(config.NOTE_DIR).resolve())
        return str(relative) if str(relative) != "." else "(top level)"
    except ValueError:
        return str(folder)


def format_event_time(value: Optional[datetime]) -> str:
    """Human-readable event time, e.g. "Wed 18 Nov 2026, 16:00"; "" for None."""
    # TODO(person 4): say "today"/"tomorrow" like notifier.build_reminder does.
    return value.strftime("%a %d %b %Y, %H:%M") if isinstance(value, datetime) else ""
