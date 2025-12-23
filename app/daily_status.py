from __future__ import annotations

from datetime import date
from flask import session

from app.escape.core import get_today_key
from app.roulette.routes import _ROULETTE_PLAY_KEY, _local_today

ESCAPE_COMPLETE_KEY = "rr_escape_complete"
BRIEF_SEEN_KEY = "rr_brief_seen"


def _safe_match(key: str, expected: str | None) -> bool:
    if not expected:
        return False
    try:
        return session.get(key) == expected
    except Exception:
        return False


def get_today_status() -> dict:
    """
    Return per-experience completion for the current visitor/session.
    Keys: escape, roulette, brief (booleans).
    """
    today_iso = date.today().isoformat()
    escape_today = get_today_key()
    roulette_today = _local_today().isoformat()

    return {
        "escape": _safe_match(ESCAPE_COMPLETE_KEY, escape_today),
        "roulette": _safe_match(_ROULETTE_PLAY_KEY, roulette_today),
        "brief": _safe_match(BRIEF_SEEN_KEY, today_iso),
    }
