# =============================================================================
# scripts/try_rag.py - owner: Person 3 (LLM RAG & Guardrails)
# Receives : nothing. Reads real files from sample_notes/ and talks to the LOCAL
#            Ollama server (config.OLLAMA_HOST) only.
# Hands off: a printed report of what the model really replies, which citations
#            were extracted, and the final AnswerDict, for the demo questions.
# Uses     : module_rag only. ContextItems are built by hand, so this does not
#            depend on Person 1's parser or Person 2's store.
# Run      : python scripts/try_rag.py         (OLLAMA_MODEL=... to try another model)
# =============================================================================
"""See how the local model behaves on the demo conversation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402  - first project import: offline env vars

import logging  # noqa: E402
from typing import Callable  # noqa: E402

from contracts import FALLBACK_MESSAGE, AnswerDict, ContextItem  # noqa: E402
from module_rag import engine, guardrails, llm_client, pipeline  # noqa: E402

NOTES = ROOT / "sample_notes"
CALL = NOTES / "meetings" / "2026-11" / "call_prep.txt"
QUILLMONT = NOTES / "Work" / "Clients" / "Quillmont" / "account_overview.md"
FORMAT = NOTES / "Desktop dump" / "untitled 3.txt"
Q2_REVIEW = NOTES / "old_notes" / "2026" / "q2_review.md"
NORRBRIDGE = NOTES / "Work" / "Clients" / "Norrbridge" / "kickoff.md"


def context_item(path: Path) -> ContextItem:
    text = path.read_text(encoding="utf-8")
    first_line = next((l.strip("# ").strip() for l in text.splitlines() if l.strip()), path.stem)
    return ContextItem(filepath=str(path.resolve()), title=first_line, content=text,
                       score=0.8, via="semantic")


Check = Callable[[AnswerDict], bool]


def cites(result: AnswerDict, *names: str) -> bool:
    cited = {Path(s).name for s in result["sources"]}
    return all(n in cited for n in names)


CONVERSATION: list[tuple[str, Check, str]] = [
    ("When is my call with the clients?",
     lambda r: r["grounded"] and "4" in r["answer"] and cites(r, "call_prep.txt"),
     "grounded, mentions 4pm, cites call_prep.txt"),
    ("Which client?",
     lambda r: r["grounded"] and "quillmont" in r["answer"].lower() and cites(r, "account_overview.md"),
     "grounded, names Quillmont, cites account_overview.md"),
    ("Is it virtual or in person?",
     lambda r: r["grounded"] and "virtual" in r["answer"].lower() and cites(r, "untitled 3.txt"),
     "grounded, says virtual, cites untitled 3.txt"),
    ("What is our parking reimbursement policy?",
     lambda r: r["answer"] == FALLBACK_MESSAGE and not r["grounded"],
     "exactly FALLBACK_MESSAGE"),
]

AMBIGUOUS: tuple[str, Check, str] = (
    "What did the client ask for?",
    lambda r: (r["grounded"] and "?" in r["answer"]
               and "quillmont" in r["answer"].lower() and "norrbridge" in r["answer"].lower()
               and cites(r, "kickoff.md")
               and any(cites(r, n) for n in ("account_overview.md", "q2_review.md"))),
    "clarifying question naming both clients, citing both clients' notes",
)

REWRITE_FOR = {"Which client?", "Is it virtual or in person?"}


class Recorder:
    """Wrap llm_client.chat to capture every raw reply before validation."""

    def __init__(self) -> None:
        self.real_chat = llm_client.chat
        self.replies: list[str] = []

    def __call__(self, messages, temperature=0.0):
        reply = self.real_chat(messages, temperature=temperature)
        self.replies.append(reply)
        return reply


def run_case(n: int, question: str, check: Check, expected: str,
             context: list[ContextItem], history: list[dict[str, str]],
             recorder: Recorder) -> tuple[AnswerDict, bool]:
    print(f"\n[{n}] Q: {question}")
    if question in REWRITE_FOR:
        print(f"    rewrite      : {pipeline.rewrite_query(question, history)!r}")
    recorder.replies.clear()
    result = engine.answer_user_query(question, context, history)
    raw = recorder.replies[-1] if recorder.replies else "(LLM not called)"
    print(f"    raw reply    : {raw!r}")
    print(f"    cited names  : {guardrails.extract_cited_filenames(raw, context)}")
    print(f"    citations ok : {guardrails.validate_citations(raw, context)}")
    print(f"    AnswerDict   : {dict(result, sources=[Path(s).name for s in result['sources']])}")
    ok = check(result)
    print(f"    expected     : {expected} -> {'PASS' if ok else 'FAIL'}")
    return result, ok


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING)

    if not llm_client.is_available():
        print(f"The local model '{config.OLLAMA_MODEL}' is not available at {config.OLLAMA_HOST}.")
        print("Start Ollama (open the Ollama app, or run: ollama serve), then pull the model once:")
        print(f"    ollama pull {config.OLLAMA_MODEL}")
        print("Or try a model you already have:  set OLLAMA_MODEL=<name>  (see: ollama list)")
        return 1

    print(f"Model: {config.OLLAMA_MODEL} at {config.OLLAMA_HOST}")
    recorder = Recorder()
    llm_client.chat = recorder  # engine and pipeline call llm_client.chat by attribute

    results: list[bool] = []
    context = [context_item(p) for p in (CALL, QUILLMONT, FORMAT)]
    print("\n=== Conversation 1 - context:", [Path(c["filepath"]).name for c in context])
    history: list[dict[str, str]] = []
    for i, (question, check, expected) in enumerate(CONVERSATION, start=1):
        result, ok = run_case(i, question, check, expected, context, history, recorder)
        results.append(ok)
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": result["answer"]}]

    context = [context_item(p) for p in (QUILLMONT, Q2_REVIEW, NORRBRIDGE)]
    print("\n=== Conversation 2 - context:", [Path(c["filepath"]).name for c in context])
    question, check, expected = AMBIGUOUS
    results.append(run_case(5, question, check, expected, context, [], recorder)[1])

    print(f"\n{sum(results)}/{len(results)} cases behaved as expected.")
    return 0 if all(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
