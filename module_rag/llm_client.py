# =============================================================================
# module_rag/llm_client.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : chat messages (list of {"role", "content"}) from engine.py and
#            pipeline.py.
# Hands off: the model's reply text (str) back to engine.py / pipeline.py;
#            is_available() is also read by Person 4's UI for a status line.
# Engine   : ollama.Client(host=config.OLLAMA_HOST), which is 127.0.0.1 only,
#            model config.OLLAMA_MODEL. No other host is ever contacted.
# =============================================================================
"""Thin wrapper around the local Ollama server."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Iterable, Optional

import config

log = logging.getLogger(__name__)

AVAILABILITY_TIMEOUT_SECONDS = 2
AVAILABILITY_CACHE_SECONDS = 30.0
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

_availability: Optional[tuple[float, bool]] = None  # (checked_at monotonic, result)
_availability_lock = threading.Lock()


def _client(timeout: float | None = None):
    """Ollama client bound to the local host. Heavy import kept inside."""
    import ollama

    return ollama.Client(
        host=config.OLLAMA_HOST,
        timeout=config.OLLAMA_TIMEOUT_SECONDS if timeout is None else timeout,
    )


def is_available() -> bool:
    """Return True if the local Ollama server answers and the model is pulled.

    Output: bool. Never raises. Uses a separate client with a 2 s timeout, and
    caches the result (True or False) for 30 s because the Streamlit sidebar
    calls this on every rerun.
    Edge cases: server down, model not pulled, ollama package missing -> False.
    """
    global _availability
    with _availability_lock:
        now = time.monotonic()
        if _availability and now - _availability[0] < AVAILABILITY_CACHE_SECONDS:
            return _availability[1]
        result = _check_available()
        _availability = (now, result)
        return result


def chat(messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    """Send ``messages`` to the local model and return the reply text.

    Inputs:
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        temperature: 0.0 by default; a fixed seed (42) keeps replies repeatable.
    Output:
        The assistant reply, stripped (any <think>...</think> block removed).
    Edge cases:
        - Server down / timeout / model missing -> raises; engine.py turns that
          into the fallback answer and pipeline.rewrite_query into "no rewrite".
        - Empty reply -> "".
    """
    response = _client().chat(
        model=config.OLLAMA_MODEL,
        messages=messages,
        options={"temperature": temperature, "seed": 42},
    )
    message = _field(response, "message")
    content = _field(message, "content") or ""
    return _THINK_RE.sub("", content).strip()


# --- helpers --------------------------------------------------------------------

def _check_available() -> bool:
    try:
        response = _client(timeout=AVAILABILITY_TIMEOUT_SECONDS).list()
        names = list(_model_names(_field(response, "models") or []))
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        log.info("Local LLM not available at %s: %s", config.OLLAMA_HOST, exc)
        return False
    if any(_same_model(config.OLLAMA_MODEL, name) for name in names):
        return True
    log.info("Model %r is not pulled. Pulled models: %s", config.OLLAMA_MODEL, names)
    return False


def _model_names(models: Iterable[Any]) -> Iterable[str]:
    """Names from list() entries: objects with .model, or dicts with "name"/"model"."""
    for entry in models:
        name = getattr(entry, "model", None)
        if not name and isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
        if name:
            yield str(name)


def _same_model(wanted: str, pulled: str) -> bool:
    """"llama3" matches "llama3:latest"; an explicit tag must match exactly."""
    wanted, pulled = wanted.strip().lower(), pulled.strip().lower()
    if ":" in wanted:
        return wanted == pulled
    return pulled.split(":", 1)[0] == wanted


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a pydantic-style response object or a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _reset_availability_cache() -> None:
    """Forget the cached is_available() result (for tests)."""
    global _availability
    with _availability_lock:
        _availability = None
