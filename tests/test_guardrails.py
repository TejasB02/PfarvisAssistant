"""The assistant must never invent: no context -> no LLM call; bad citations -> fallback."""

from unittest.mock import Mock

import pytest

import config
from contracts import FALLBACK_MESSAGE, ContextItem, fallback_answer
from module_rag import engine, guardrails, llm_client


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
