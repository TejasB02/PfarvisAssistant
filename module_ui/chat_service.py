from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence, Tuple

from contracts import AnswerDict, note_from_dict


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
        self, messages: Sequence[Message]
    ) -> ChatReply:
        ...
