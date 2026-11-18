# =============================================================================
# module_rag/llm_client.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : chat messages (list of {"role", "content"}) from engine.py and
#            pipeline.py.
# Hands off: the model's reply text (str) back to engine.py / pipeline.py;
#            is_available() is also read by Person 4's UI for a status line.
# Engine   : ollama.Client(host=config.OLLAMA_HOST), which is 127.0.0.1 only,
#            model config.OLLAMA_MODEL.
# STATUS   : STUB - no Ollama call; returns a mock reply.
# =============================================================================
"""Thin wrapper around the local Ollama server."""

from __future__ import annotations

import logging
import re

import config
from contracts import FALLBACK_MESSAGE

log = logging.getLogger(__name__)


def _client():
    """Create the Ollama client bound to the local host (import inside)."""
    import ollama  # heavy import kept inside the function

    return ollama.Client(host=config.OLLAMA_HOST, timeout=config.LLM_TIMEOUT_SECONDS)


def is_available() -> bool:
    """Return True if the local Ollama server answers and the model is pulled.

    Output: bool. Must never raise and must return quickly (short timeout),
    because the UI calls it on every rerun.
    Edge cases: server down, model not pulled, ollama package missing -> False.
    """
    # TODO(person 3): real check, e.g.
    #   try: models = _client().list(); return any(config.OLLAMA_MODEL in m.model ...)
    #   except Exception: return False
    return True  # mock: pretend the local model is up


def chat(messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    """Send ``messages`` to the local model and return the reply text.

    Inputs:
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        temperature: 0.0 by default for repeatable, grounded answers.
    Output:
        The assistant reply, stripped.
    Edge cases:
        - Server down / timeout -> raise; engine.py catches and returns the
          fallback answer.
        - Empty reply -> "".
    """
    # TODO(person 3): real call:
    #   resp = _client().chat(model=config.OLLAMA_MODEL, messages=messages,
    #                         options={"temperature": temperature})
    #   return resp["message"]["content"].strip()
    # Mock: cite the first SOURCE header from the CONTEXT block (see engine.py).
    last = messages[-1]["content"] if messages else ""
    match = re.search(r"^SOURCE \[([^\]\n]+)\]", last, flags=re.MULTILINE)
    if not match:
        return FALLBACK_MESSAGE
    return (
        "(mock reply - the local LLM is not wired up yet) "
        f"The most relevant note is [{match.group(1)}]."
    )
