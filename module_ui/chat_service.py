from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence, Tuple

import contracts
from contracts import AnswerDict, FALLBACK_MESSAGE, note_from_dict


@dataclass(frozen=True)
class SourceReference:
    title: str
    location: str = ""
    excerpt: str = ""


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str
    sources: Tuple[SourceReference, ...] = ()
    grounded: Optional[bool] = None


@dataclass(frozen=True)
class ChatReply:
    text: str
    sources: Tuple[SourceReference, ...] = ()
    grounded: Optional[bool] = None

    @classmethod
    def from_answer(
        cls, answer: AnswerDict, notes: Sequence[dict[str, Any]]
    ) -> "ChatReply":
        notes_by_path = {note["filepath"]: note for note in notes}
        references = []
        for filepath in answer["sources"]:
            note_data = notes_by_path.get(filepath)
            if note_data is None:
                references.append(SourceReference(Path(filepath).name, filepath))
                continue
            note = note_from_dict(note_data)
            references.append(SourceReference(
                note.title or Path(filepath).name, filepath, note.content
            ))
        return cls(answer["answer"], tuple(references), answer["grounded"])


class ChatService(Protocol):
    def reply(
        self, messages: Sequence[Message], source_types: Optional[list[str]] = None
    ) -> ChatReply:
        ...


class MockChatService:
    def reply(
        self, messages: Sequence[Message], source_types: Optional[list[str]] = None
    ) -> ChatReply:
        source_catalog = getattr(contracts, "SOURCE_TYPES", {})
        selected = [kind for kind in source_catalog if kind in (source_types or [])]
        hint = ""
        if selected and len(selected) < len(source_catalog):
            labels = ", ".join(source_catalog[kind]["label"] for kind in selected)
            hint = f"\n\n(Searched only: {labels}.)"
        if selected and contracts.source_type_for("whatsapp_export.txt") not in selected:
            return ChatReply(FALLBACK_MESSAGE + hint)
        if messages and "event" in messages[-1].text.casefold():
            return ChatReply(
                "Saturday’s event is at **Riverside Community Hall** [1]. "
                "Setup starts at **9:00 am** and guests arrive at **10:00 am** [2]. "
                "The remaining tasks are to confirm the projector and print the sign-in sheet [3]."
                + hint,
                sources=(
                    SourceReference(
                        "whatsapp_export.txt",
                        excerpt="Venue confirmed: Riverside Community Hall.",
                    ),
                    SourceReference(
                        "event_schedule.txt",
                        excerpt="Saturday: setup at 9:00 am, guests arrive at 10:00 am.",
                    ),
                    SourceReference(
                        "planning_notes.txt",
                        excerpt="To do: confirm the projector and print the sign-in sheet.",
                    ),
                ),
            )
        return ChatReply(
            "The mock demo has sample information about Saturday’s event. "
            "Try asking: What’s the plan for Saturday’s event, and what’s still outstanding?"
            + hint
        )
