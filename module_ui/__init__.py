# =============================================================================
# module_ui - Person 4: Frontend, State & Notifications (also owns app.py)
# Receives : note dicts from Person 1, the NotesStore from Person 2, and
#            AnswerDicts from Person 3 (via module_rag.pipeline.ask).
# Hands off: the Streamlit chat + sidebar the manager sees, desktop
#            notifications, and pinned note ids back into pipeline.ask.
# Files    : ui_app.py (Streamlit page), notifier.py (background reminders).
# =============================================================================
"""UI package: Streamlit chat, notes sidebar and reminders."""
