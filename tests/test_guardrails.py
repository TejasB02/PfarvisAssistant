"""The assistant must never invent: no context -> no LLM call; bad citations -> fallback."""

from pathlib import Path
from unittest.mock import Mock

import pytest

import config
from contracts import FALLBACK_MESSAGE, ContextItem, fallback_answer
from module_rag import engine, guardrails, llm_client, pipeline

ROOT = Path(__file__).resolve().parents[1]


def _ctx(name: str, score: float = 0.9, via: str = "semantic") -> ContextItem:
    return ContextItem(
        filepath=f"/notes/meetings/{name}", title=name, content=f"content of {name}",
        score=score, via=via,
    )


@pytest.fixture
def fake_chat(monkeypatch):
    mock = Mock(return_value="The call is at 4pm [call_prep.txt].")
    monkeypatch.setattr(llm_client, "chat", mock)
    return mock


def test_empty_context_returns_fallback_without_calling_llm(fake_chat):
    result = engine.answer_user_query("When is my call?", [], [])
    assert result == fallback_answer()
    fake_chat.assert_not_called()


def test_low_score_context_returns_fallback_without_calling_llm(fake_chat):
    weak = [_ctx("call_prep.txt", score=config.MIN_SIMILARITY / 2)]
    assert engine.answer_user_query("When is my call?", weak, []) == fallback_answer()
    fake_chat.assert_not_called()


def test_citation_of_file_not_in_context_fails():
    context = [_ctx("call_prep.txt")]
    assert guardrails.validate_citations("It is virtual [untitled 3.txt].", context) is False


def test_answer_without_any_citation_fails():
    assert guardrails.validate_citations("It is at 4pm.", [_ctx("call_prep.txt")]) is False


def test_valid_citation_passes():
    context = [_ctx("call_prep.txt"), _ctx("untitled 3.txt")]
    answer = "At 4pm [call_prep.txt], and it is virtual [untitled 3.txt]."
    assert guardrails.validate_citations(answer, context) is True


def test_engine_rejects_hallucinated_citation(fake_chat):
    fake_chat.return_value = "It is with Acme [invented.md]."
    result = engine.answer_user_query("Which client?", [_ctx("call_prep.txt")], [])
    assert result == fallback_answer()
    fake_chat.assert_called_once()


def test_engine_returns_grounded_answer_with_sources(fake_chat):
    context = [_ctx("call_prep.txt"), _ctx("other.md", score=0.5, via="graph")]
    result = engine.answer_user_query("When is my call?", context, [])
    assert result["grounded"] is True
    assert result["sources"] == ["/notes/meetings/call_prep.txt"]


def test_engine_passes_through_model_fallback(fake_chat):
    fake_chat.return_value = FALLBACK_MESSAGE
    assert engine.answer_user_query("Parking policy?", [_ctx("a.md")], []) == fallback_answer()


def test_engine_survives_llm_errors(fake_chat):
    fake_chat.side_effect = ConnectionError("ollama is not running")
    assert engine.answer_user_query("When?", [_ctx("a.md")], []) == fallback_answer()


def test_system_prompt_states_the_rules():
    prompt = guardrails.SYSTEM_PROMPT
    assert FALLBACK_MESSAGE in prompt
    assert "CONTEXT" in prompt and "square brackets" in prompt


# --- gate: only semantic hits can open it -----------------------------------------

def test_graph_or_pinned_items_alone_cannot_open_the_gate():
    assert guardrails.has_enough_context([_ctx("a.md", score=1.0, via="graph")]) is False
    assert guardrails.has_enough_context([_ctx("a.md", score=0.0, via="graph"),
                                          _ctx("b.md", score=0.9, via="semantic")]) is True


# --- citation formats ---------------------------------------------------------------

CITATION_CONTEXT = [_ctx(n) for n in ("call_prep.txt", "account_overview.md", "q2_review.md", "kickoff.md")]

# Replies the local model actually produced in scripts/try_rag.py (gemma4:e4b).
SEEN_FORMATS = [
    "The call with the clients is scheduled for 4pm today [call_prep.txt].",
    "Which client are you referring to: Quillmont Systems [account_overview.md, q2_review.md] "
    "or Norrbridge Foods [kickoff.md]?",
]
# Shapes local models commonly produce; accepted so a correct answer is not thrown away.
ANTICIPATED_FORMATS = [
    "The call is at 4pm [call_prep].",
    "The call is at 4pm (call_prep.txt).",
    "The call is at 4pm [meetings/2026-11/call_prep.txt].",
    "The call is at 4pm [C:\\notes\\meetings\\2026-11\\call_prep.txt].",
    "The call is at 4pm [Source: CALL_PREP.TXT].",
]


@pytest.mark.parametrize("answer", SEEN_FORMATS + ANTICIPATED_FORMATS)
def test_citation_formats_accepted_when_file_in_context(answer):
    assert guardrails.validate_citations(answer, CITATION_CONTEXT) is True
    assert guardrails.cited_sources(answer, CITATION_CONTEXT)


@pytest.mark.parametrize("answer", SEEN_FORMATS + ANTICIPATED_FORMATS)
def test_citation_formats_rejected_when_file_not_in_context(answer):
    assert guardrails.validate_citations(answer, [_ctx("other_note.md")]) is False


def test_one_invented_file_among_real_ones_fails():
    answer = "At 4pm [call_prep.txt] with Quillmont [invented.md]."
    assert guardrails.validate_citations(answer, CITATION_CONTEXT) is False


def test_plain_parentheses_are_not_citations():
    answer = "It is virtual (not in person) [call_prep.txt]."
    assert guardrails.extract_cited_filenames(answer, CITATION_CONTEXT) == ["call_prep.txt"]
    assert guardrails.validate_citations(answer, CITATION_CONTEXT) is True


# --- rewrite_query --------------------------------------------------------------------

HISTORY = [
    {"role": "user", "content": "When is my call with the clients?"},
    {"role": "assistant", "content": "The call is at 4pm today [call_prep.txt]."},
]


def test_rewrite_with_empty_history_makes_no_call(fake_chat):
    assert pipeline.rewrite_query("Which client?", []) == "Which client?"
    fake_chat.assert_not_called()


def test_rewrite_returns_good_rewrite(fake_chat):
    fake_chat.return_value = "Standalone question: Which client is the 4pm call today with?"
    assert pipeline.rewrite_query("Which client?", HISTORY) == "Which client is the 4pm call today with?"
    fake_chat.assert_called_once()
    assert fake_chat.call_args.kwargs.get("temperature", 0.0) == 0.0


@pytest.mark.parametrize("reply", ["", "   ", "x" * (3 * len("Which client?") + 201)])
def test_rewrite_rejects_empty_or_overlong_output(fake_chat, reply):
    fake_chat.return_value = reply
    assert pipeline.rewrite_query("Which client?", HISTORY) == "Which client?"


def test_rewrite_survives_llm_error(fake_chat):
    fake_chat.side_effect = ConnectionError("ollama is not running")
    assert pipeline.rewrite_query("Which client?", HISTORY) == "Which client?"


# --- pipeline.ask with a fake store -----------------------------------------------------

class FakeStore:
    def __init__(self, hits: list[ContextItem], notes: dict[str, dict]):
        self.hits, self.notes, self.queries = hits, notes, []

    def hybrid_search(self, query, k=5, hops=1):
        self.queries.append(query)
        return [dict(h) for h in self.hits]

    def get_note(self, note_id):
        return self.notes.get(note_id)


def _note(note_id: str, name: str) -> dict:
    return {"id": note_id, "title": name, "content": f"content of {name}",
            "filepath": f"/notes/meetings/{name}", "timestamp": None, "client": None,
            "tags": [], "event_time": None}


@pytest.fixture
def captured_context(monkeypatch):
    seen: list[list[ContextItem]] = []

    def fake_answer(query, retrieved_context, chat_history):
        seen.append(retrieved_context)
        return fallback_answer()

    monkeypatch.setattr(engine, "answer_user_query", fake_answer)
    return seen


def test_ask_adds_pinned_notes_with_zero_score(captured_context, fake_chat):
    store = FakeStore([_ctx("call_prep.txt")], {"p1": _note("p1", "untitled 3.txt")})
    pipeline.ask("Is it virtual?", [], store, pinned_note_ids=["p1"])
    context = captured_context[0]
    assert [Path(c["filepath"]).name for c in context] == ["call_prep.txt", "untitled 3.txt"]
    assert context[1]["score"] == 0.0 and context[1]["via"] == "graph"


def test_ask_ignores_missing_pinned_ids(captured_context, fake_chat):
    store = FakeStore([_ctx("call_prep.txt")], {})
    pipeline.ask("Is it virtual?", [], store, pinned_note_ids=["gone"])
    assert len(captured_context[0]) == 1


def test_ask_does_not_add_a_pinned_note_twice(captured_context, fake_chat):
    store = FakeStore([_ctx("call_prep.txt")], {"p1": _note("p1", "call_prep.txt")})
    pipeline.ask("When?", [], store, pinned_note_ids=["p1", "p1"])
    assert [c["filepath"] for c in captured_context[0]] == ["/notes/meetings/call_prep.txt"]


def test_ask_with_only_pinned_notes_returns_fallback_without_llm(fake_chat):
    store = FakeStore([], {"p1": _note("p1", "untitled 3.txt")})
    assert pipeline.ask("Is it virtual?", [], store, pinned_note_ids=["p1"]) == fallback_answer()
    fake_chat.assert_not_called()


def test_ask_searches_with_the_rewritten_query(captured_context, fake_chat):
    fake_chat.return_value = "Which client is the 4pm call today with?"
    store = FakeStore([_ctx("call_prep.txt")], {})
    pipeline.ask("Which client?", HISTORY, store)
    assert store.queries == ["Which client is the 4pm call today with?"]


# --- live check against the real local model (skipped when it is not available) -------

@pytest.mark.skipif(not llm_client.is_available(), reason="local Ollama model not available")
def test_live_model_returns_fallback_for_topic_not_in_notes():
    notes = ROOT / "sample_notes"
    paths = [notes / "meetings" / "2026-11" / "call_prep.txt",
             notes / "Work" / "Clients" / "Quillmont" / "account_overview.md",
             notes / "Desktop dump" / "untitled 3.txt"]
    context = [ContextItem(filepath=str(p), title=p.stem, content=p.read_text(encoding="utf-8"),
                           score=0.8, via="semantic") for p in paths]
    result = engine.answer_user_query("What is our parking reimbursement policy?", context, [])
    assert result["answer"] == FALLBACK_MESSAGE
