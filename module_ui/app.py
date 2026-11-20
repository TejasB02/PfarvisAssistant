from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

import streamlit as st

from module_ui.chat_service import ChatService, Message
from module_ui.voice import SAMPLE_RATE, transcribe

log = logging.getLogger(__name__)


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)

    @property
    def title(self) -> str:
        if not self.messages:
            return "Untitled chat"
        title = " ".join(self.messages[0].text.split())
        return title if len(title) <= 32 else title[:31].rstrip() + "…"


def start_chat() -> None:
    st.session_state.active_chat = None


def select_chat(index: int) -> None:
    st.session_state.active_chat = index


def delete_chat(index: int) -> None:
    active_index = st.session_state.active_chat
    st.session_state.conversations.pop(index)
    if active_index == index:
        start_chat()
    elif active_index is not None and active_index > index:
        select_chat(active_index - 1)


def add_messages(*messages: Message) -> None:
    conversations = st.session_state.conversations
    if st.session_state.active_chat is None:
        conversations.append(Conversation())
        select_chat(len(conversations) - 1)
    conversations[st.session_state.active_chat].messages.extend(messages)


def main(
    service: ChatService,
    *,
    status: str,
    render_notes: Callable[[], None] | None = None,
    render_reminders: Callable[[], None] | None = None,
) -> None:
    st.set_page_config(page_title="Pfarvis Assistant", page_icon="✦")
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    if "conversations" not in st.session_state:
        st.session_state.conversations = []
        start_chat()

    conversations: list[Conversation] = st.session_state.conversations
    active_index = st.session_state.active_chat

    with st.sidebar:
        st.subheader("Pfarvis Assistant")
        st.button("＋ New chat", type="primary", on_click=start_chat)
        st.divider()
        st.caption("CHAT HISTORY")
        if not conversations:
            st.caption("Your conversations will appear here.")
        for index in reversed(range(len(conversations))):
            title = conversations[index].title
            chat_column, delete_column = st.columns([4, 1], gap="small")
            chat_column.button(
                title,
                key=f"chat-{index}",
                on_click=select_chat,
                args=(index,),
                type="primary" if index == active_index else "secondary",
            )
            delete_column.button(
                f"Delete {title}",
                icon=":material/delete:",
                key=f"delete-chat-{index}",
                help=f"Delete chat: {title}",
                on_click=delete_chat,
                args=(index,),
            )

    if render_notes is not None:
        render_notes()

    st.title("Pfarvis Assistant", anchor=False)
    st.caption(status)
    if render_reminders is not None:
        render_reminders()
    chat = st.container()
    welcome = chat.empty()
    if active_index is None:
        welcome.subheader("What can I help you explore?", anchor=False)
    else:
        for message in conversations[active_index].messages:
            with chat.chat_message(message.role):
                if message.role == "user":
                    st.text(message.text)
                else:
                    st.markdown(message.text)
                    for number, source in enumerate(message.sources, start=1):
                        with st.expander(f"[{number}] {source.title}"):
                            if source.location:
                                st.caption(source.location)
                            if source.excerpt:
                                st.text(source.excerpt)

    message_column, voice_column = st.columns([5, 1], vertical_alignment="center")
    with voice_column.popover("Voice", icon=":material/mic:"):
        st.caption("Record your question, then review the text before sending.")
        recording = st.audio_input("Record question", sample_rate=SAMPLE_RATE)
        if recording and recording.file_id != st.session_state.get("transcribed_recording"):
            st.session_state.transcribed_recording = recording.file_id
            try:
                with st.spinner("Transcribing locally…"):
                    transcript = transcribe(recording.getvalue())
                if transcript:
                    st.session_state["query_input"] = transcript
                else:
                    st.warning("No speech detected. Try recording again.")
            except FileNotFoundError:
                st.error("The local Whisper model is missing. Voice input needs the model installed.")
            except Exception:
                log.exception("Could not transcribe the recording")
                st.error("Could not transcribe the recording. Please try again.")

    prompt = message_column.chat_input("Message Pfarvis Assistant", key="query_input")

    if prompt and prompt.strip():
        welcome.empty()
        message = Message(role="user", text=prompt.strip())
        history = () if active_index is None else conversations[active_index].messages
        messages = (*history, message)
        with chat.chat_message("user"):
            st.text(message.text)
        with chat.chat_message("assistant"):
            try:
                with st.spinner("Thinking…", show_time=True):
                    reply = service.reply(messages)
            except Exception:
                log.exception("Could not complete the chat request")
                st.error("Something went wrong. Please try sending your message again.")
                return
        add_messages(message, Message("assistant", reply.text, reply.sources, reply.grounded))
        st.rerun()
