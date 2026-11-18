# =============================================================================
# module_ui/notifier.py - owner: Person 4 (Frontend, State & Notifications)
# Receives : a get_notes() callable (NotesStore.all_notes, Person 2) whose
#            dicts carry event_time (extracted by Person 1).
# Hands off: desktop notifications (plyer) and reminder dicts on a
#            thread-safe queue -> ui_app.reminder_panel (Person 4) via
#            pop_pending_reminders().
# RULE     : the scheduler thread NEVER touches st.session_state and never
#            imports streamlit. It only writes to the queue.
# STATUS   : minimal working version so the reminder demo runs; harden it.
# =============================================================================
"""Background reminders: check event times every N seconds, fire once per note."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Callable, Optional

import config

log = logging.getLogger(__name__)

_pending: "queue.Queue[dict]" = queue.Queue()
_fired_ids: set[str] = set()
_fired_lock = threading.Lock()
_start_lock = threading.Lock()
_scheduler: object | None = None  # BackgroundScheduler, or a fallback Thread


def start_scheduler(get_notes: Callable[[], list[dict]]) -> None:
    """Start the background reminder check (idempotent: one per process).

    Inputs:
        get_notes: returns the current note dicts (e.g. store.all_notes).
    Output: None. Returns immediately.
    Behaviour:
        - APScheduler BackgroundScheduler, interval job every
          config.SCHEDULER_INTERVAL_SECONDS (first run immediately).
        - A note fires once when event_time falls within the next
          config.REMINDER_LEAD_MINUTES; fired ids are remembered so nothing repeats.
        - If apscheduler is not installed yet, falls back to a daemon thread
          so the scaffold still demos with only streamlit installed.
    """
    # TODO(person 4): add stop/reset for tests, snooze, and re-fire when a
    # note's event_time changes (key fired ids by (id, event_time)?).
    global _scheduler
    with _start_lock:
        if _scheduler is not None:
            return
        interval = config.SCHEDULER_INTERVAL_SECONDS
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            log.warning("apscheduler not installed; using a simple reminder thread")
            thread = threading.Thread(
                target=_fallback_loop, args=(get_notes, interval),
                name="reminder-loop", daemon=True,
            )
            thread.start()
            _scheduler = thread
            return
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            check_due_reminders, "interval", seconds=interval, args=[get_notes],
            id="reminders", next_run_time=datetime.now(),
            max_instances=1, coalesce=True,
        )
        scheduler.start()
        _scheduler = scheduler


def check_due_reminders(
    get_notes: Callable[[], list[dict]],
    now: Optional[datetime] = None,
) -> list[dict]:
    """One scheduler tick: queue + notify every note that is due. Returns them.

    Due means ``now <= event_time <= now + REMINDER_LEAD_MINUTES`` and the
    note id has not fired before. Never raises (a crash would kill the job).
    """
    # TODO(person 4): decide what to do with events missed while the app was closed.
    now = now or datetime.now()
    window_end = now + timedelta(minutes=config.REMINDER_LEAD_MINUTES)
    fired: list[dict] = []
    try:
        notes = get_notes()
    except Exception:  # noqa: BLE001
        log.exception("get_notes failed in reminder check")
        return fired
    for note in notes:
        event_time = note.get("event_time")
        if not isinstance(event_time, datetime):
            continue
        if event_time.tzinfo is not None:  # contract says naive local; be lenient
            event_time = event_time.astimezone().replace(tzinfo=None)
        if not (now <= event_time <= window_end):
            continue
        with _fired_lock:
            if note["id"] in _fired_ids:
                continue
            _fired_ids.add(note["id"])
        reminder = build_reminder(note, event_time, now)
        _pending.put(reminder)
        notify_desktop("Local Notes Assistant", reminder["message"])
        fired.append(reminder)
    return fired


def build_reminder(note: dict, event_time: datetime, now: datetime) -> dict:
    """Reminder dict consumed by ui_app: note_id, title, filepath, filename,
    event_time, message, content."""
    if event_time.date() == now.date():
        when = f"today at {event_time:%H:%M}"
    elif event_time.date() == (now + timedelta(days=1)).date():
        when = f"tomorrow at {event_time:%H:%M}"
    else:
        when = f"{event_time:%a %d %b at %H:%M}"
    return {
        "note_id": note["id"],
        "title": note["title"],
        "filepath": note["filepath"],
        "filename": PurePosixPath(note["filepath"].replace("\\", "/")).name,
        "event_time": event_time,
        "message": f"Coming up {when}: {note['title']}",
        "content": note.get("content", ""),
    }


def notify_desktop(title: str, message: str) -> None:
    """Show a local desktop notification. Never raises.

    Uses plyer (imported inside); backends differ per OS, so every failure is
    logged and swallowed. Desktop notifications cannot carry buttons; the
    "Add to chat" button lives in the Streamlit reminder card instead.
    """
    # TODO(person 4): test on each teammate's OS; plyer on Windows truncates ~256 chars.
    try:
        from plyer import notification

        notification.notify(
            title=title[:63], message=message[:250],
            app_name="Local Notes Assistant", timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        log.info("Desktop notification unavailable (%s): %s", type(exc).__name__, message)


def pop_pending_reminders() -> list[dict]:
    """Drain and return all queued reminder dicts (oldest first). Thread-safe.

    Called by the UI fragment on the Streamlit thread. Returns [] when empty.
    The queue is per process, so with several browser tabs open only one tab
    receives each reminder.
    """
    # TODO(person 4): keep this non-blocking.
    items: list[dict] = []
    while True:
        try:
            items.append(_pending.get_nowait())
        except queue.Empty:
            return items


def _fallback_loop(get_notes: Callable[[], list[dict]], interval: int) -> None:
    while True:
        check_due_reminders(get_notes)
        time.sleep(interval)
