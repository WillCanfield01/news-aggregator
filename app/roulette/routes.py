# app/roulette/routes.py
from __future__ import annotations

import hashlib
import os
import random
from datetime import datetime, timedelta, date
from pathlib import Path

import pytz
from flask import (
    abort,
    current_app,
    jsonify,
    make_response,
    render_template,
    request,
    url_for,
)
from sqlalchemy import and_, func

from app.extensions import db

# Optional: support without flask-login present
try:
    from flask_login import current_user  # type: ignore
except Exception:  # pragma: no cover
    class _Anon:
        is_authenticated = False
        id = None
    current_user = _Anon()  # type: ignore

from . import roulette_bp
from .models import TimelineGuess, TimelineRound, TimelineStreak


# -----------------------------------------------------------------------------
# Timezone: make routes use the SAME local "today" as the generator does
# -----------------------------------------------------------------------------
_TZ_NAME = os.getenv("TIME_ZONE", "America/Denver")
_TZ = pytz.timezone(_TZ_NAME)


def _local_today() -> date:
    """Return today's date in the configured local timezone."""
    return datetime.now(_TZ).date()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _anonymize_ip(ip: str | None) -> str:
    if not ip:
        return "unknown"
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:40]


def _today_round_or_404() -> TimelineRound:
    """Fetch today's round (local TZ) or 404 with a friendly message."""
    r = TimelineRound.query.filter_by(round_date=_local_today()).first()
    if not r:
        abort(404, description="No round generated for today.")
    return r


def _icon_url_or_fallback(name: str | None) -> str:
    """
    Resolve a roulette icon URL from /static/roulette/icons.
    Falls back to the first available SVG if the requested name doesn't exist.
    """
    static_dir = Path(current_app.static_folder) / "roulette" / "icons"

    # preferred fallbacks in order
    ordered_fallbacks = [
        "star.svg",
        "asterisk.svg",
        "dot.svg",
        "circle.svg",
        "history.svg",
        "compass.svg",
        "feather.svg",
    ]
    fallback = None
    for fb in ordered_fallbacks:
        if (static_dir / fb).exists():
            fallback = fb
            break
    if not fallback:
        # pick any svg if the folder isn't empty
        any_svg = next((p.name for p in static_dir.glob("*.svg")), None)
        fallback = any_svg or "star.svg"

    if not name:
        return url_for("static", filename=f"roulette/icons/{fallback}")
    if (static_dir / name).exists():
        return url_for("static", filename=f"roulette/icons/{name}")
    return url_for("static", filename=f"roulette/icons/{fallback}")


def _update_server_streak_if_logged_in() -> int | None:
    """Update a DB-backed streak for logged in users; return the streak or None."""
    if not getattr(current_user, "is_authenticated", False):
        return None

    today = _local_today()
    streak = TimelineStreak.query.filter_by(user_id=current_user.id).first()

    if not streak:
        streak = TimelineStreak(user_id=current_user.id, current_streak=1, last_play_date=today)
        db.session.add(streak)
        db.session.commit()
        return streak.current_streak

    if streak.last_play_date == today:
        return streak.current_streak

    if streak.last_play_date == today - timedelta(days=1):
        streak.current_streak += 1
    else:
        streak.current_streak = 1

    streak.last_play_date = today
    db.session.commit()
    return streak.current_streak


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
@roulette_bp.get("/roulette")
def play_today():
    r = _today_round_or_404()

    # Build shuffled card set; remember where the real one lands (index 0 == real)
    def _image_or_icon(url: str | None, icon_name: str | None) -> str:
        if url:
            return url
        return _icon_url_or_fallback(icon_name)

    cards = [
        {
            "orig_idx": 0,
            "text": r.real_title,
            "label": "A",
            "img": _image_or_icon(getattr(r, "real_img_url", None), r.real_icon),
        },
        {
            "orig_idx": 1,
            "text": r.fake1_title,
            "label": "B",
            "img": _image_or_icon(getattr(r, "fake1_img_url", None), r.fake1_icon),
        },
        {
            "orig_idx": 2,
            "text": r.fake2_title,
            "label": "C",
            "img": _image_or_icon(getattr(r, "fake2_img_url", None), r.fake2_icon),
        },
    ]

    random.shuffle(cards)
    correct_shuffled_idx = next(i for i, c in enumerate(cards) if c["orig_idx"] == 0)

    # Stats for today
    total = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(TimelineGuess.round_id == r.id)
        .scalar()
        or 0
    )
    correct = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True)))
        .scalar()
        or 0
    )

    # cookie guest streak (DB streak handled when guess posts)
    guest_streak = request.cookies.get("tr_streak")

    # attribution (if using Unsplash elsewhere)
    img_attr = r.real_img_attr or r.fake1_img_attr or r.fake2_img_attr

    return render_template(
        "roulette/play.html",
        round_date=r.round_date,
        cards=cards,
        correct_index=correct_shuffled_idx,
        source_url=r.real_source_url,
        day_total=total,
        day_correct=correct,
        guest_streak=guest_streak,
        img_attr=img_attr,
    )


@roulette_bp.post("/roulette/guess")
def submit_guess():
    r = _today_round_or_404()

    payload = request.get_json(force=True) or {}
    try:
        choice = int(payload.get("choice", -1))
        correct_index = int(payload.get("correct_index", -1))
    except Exception:
        return jsonify({"ok": False, "error": "bad_input"}), 400

    if choice not in (0, 1, 2) or correct_index not in (0, 1, 2):
        return jsonify({"ok": False, "error": "bad_input"}), 400

    is_correct = (choice == correct_index)
    ip_h = _anonymize_ip(request.headers.get("X-Forwarded-For", request.remote_addr))

    guess = TimelineGuess(
        round_id=r.id,
        user_id=getattr(current_user, "id", None) if getattr(current_user, "is_authenticated", False) else None,
        choice_index=choice,
        is_correct=is_correct,
        ip_hash=ip_h,
    )
    db.session.add(guess)
    db.session.commit()

    # update server streak for logged-in users; guests use cookie streak below
    server_streak = _update_server_streak_if_logged_in()

    # guest cookie streak
    today_iso = _local_today().isoformat()
    last = request.cookies.get("tr_last_play_date")
    guest_streak = int(request.cookies.get("tr_streak", "0"))
    if last != today_iso:
        guest_streak = guest_streak + 1 if last == (_local_today() - timedelta(days=1)).isoformat() else 1

    # updated totals
    total = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(TimelineGuess.round_id == r.id)
        .scalar()
        or 0
    )
    correct_ct = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True)))
        .scalar()
        or 0
    )

    resp = make_response(
        jsonify(
            {
                "ok": True,
                "correct": is_correct,
                "totals": {"plays": total, "correct": correct_ct},
                "streak": server_streak if server_streak is not None else guest_streak,
            }
        )
    )
    # Update guest streak cookies
    resp.set_cookie("tr_streak", str(guest_streak), max_age=7 * 24 * 3600, httponly=False, samesite="Lax")
    resp.set_cookie("tr_last_play_date", today_iso, max_age=7 * 24 * 3600, httponly=False, samesite="Lax")
    return resp


@roulette_bp.get("/roulette/leaderboard")
def leaderboard():
    r = _today_round_or_404()

    # Today totals
    today_total = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(TimelineGuess.round_id == r.id)
        .scalar()
        or 0
    )
    today_correct = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True)))
        .scalar()
        or 0
    )

    # Past 7 days (local timezone dates)
    seven_days_ago = _local_today() - timedelta(days=6)
    rounds = db.session.query(TimelineRound.id).filter(TimelineRound.round_date >= seven_days_ago).subquery()
    week_total = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(TimelineGuess.round_id.in_(rounds))
        .scalar()
        or 0
    )
    week_correct = (
        db.session.query(func.count(TimelineGuess.id))
        .filter(and_(TimelineGuess.round_id.in_(rounds), TimelineGuess.is_correct.is_(True)))
        .scalar()
        or 0
    )

    return render_template(
        "roulette/leaderboard.html",
        today_total=today_total,
        today_correct=today_correct,
        week_total=week_total,
        week_correct=week_correct,
    )


# -----------------------------------------------------------------------------
# Admin: hit this to force-generate today's round
# GET /roulette/admin/regen?token=...&force=1
# -----------------------------------------------------------------------------
@roulette_bp.get("/roulette/admin/regen")
def admin_regen():
    token = request.args.get("token", "")
    expected = os.getenv("ROULETTE_ADMIN_TOKEN", "")
    if not expected or token != expected:
        # Hide existence of this route
        abort(404)

    force = request.args.get("force", "0") in ("1", "true", "True", "yes")
    # Late import to avoid circular import at module load
    from app.scripts.generate_timeline_round import ensure_today_round

    ok = bool(ensure_today_round(force=force))
    return jsonify({"ok": ok, "date": _local_today().isoformat(), "tz": _TZ_NAME})
