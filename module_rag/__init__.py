# =============================================================================
# module_rag - Person 3: LLM RAG & Guardrails
# Receives : user question + chat history + NotesStore from Person 4 (UI).
# Hands off: AnswerDict {"answer", "sources", "grounded"} -> Person 4 (UI).
# Files    : pipeline.py (ask: the ONLY entry point the UI calls), engine.py
#            (answer_user_query), guardrails.py (prompt + checks),
#            llm_client.py (local Ollama at 127.0.0.1 only).
# =============================================================================
"""RAG package: grounded answers from local notes, never invented ones."""
