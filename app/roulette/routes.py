from datetime import date, timedelta
import hashlib
import random

from flask import render_template, request, jsonify, abort, make_response
from sqlalchemy import func, and_
from pathlib import Path
from flask import url_for, current_app
from app.extensions import db

# Try to use Flask-Login if present; otherwise a dummy object
try:
    from flask_login import current_user  # type: ignore
except Exception:  # pragma: no cover
    class _Anon:
        is_authenticated = False
        id = None
    current_user = _Anon()  # type: ignore

from . import roulette_bp
from .models import TimelineRound, TimelineGuess, TimelineStreak


# -------- Utilities --------
def _today_round() -> TimelineRound:
    r = TimelineRound.query.filter_by(round_date=date.today()).first()
    if not r:
        abort(404, description="No round generated for today.")
    return r


def _anonymize_ip(ip: str | None) -> str:
    if not ip:
        return "unknown"
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:40]


def _update_server_streak_if_logged_in():
    if not getattr(current_user, "is_authenticated", False):
        return None

    streak = TimelineStreak.query.filter_by(user_id=current_user.id).first()
    today = date.today()

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


# -------- Pages --------
@roulette_bp.get("/roulette")
def play_today():
    r = _today_round()


    def icon_url(name: str | None) -> str | None:
        return url_for("static", filename=f"roulette/icons/{name}") if name else None

    cards = [
        {"orig_idx": 0, "text": r.real_title, "label": "A", "icon": _icon_url_or_fallback(r.real_icon)},
        {"orig_idx": 1, "text": r.fake1_title, "label": "B", "icon": _icon_url_or_fallback(r.fake1_icon)},
        {"orig_idx": 2, "text": r.fake2_title, "label": "C", "icon": _icon_url_or_fallback(r.fake2_icon)},
    ]
    random.shuffle(cards)
    correct_shuffled_idx = next(i for i, c in enumerate(cards) if c["orig_idx"] == 0)

    total = db.session.query(func.count(TimelineGuess.id)).filter(TimelineGuess.round_id == r.id).scalar() or 0
    correct = db.session.query(func.count(TimelineGuess.id)).filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True))).scalar() or 0

    guest_streak = request.cookies.get("tr_streak")

    # pack brief attribution (optional)
    attr = r.real_img_attr or r.fake1_img_attr or r.fake2_img_attr

    return render_template(
        "roulette/play.html",
        round_date=r.round_date,
        cards=cards,
        correct_index=correct_shuffled_idx,
        source_url=r.real_source_url,
        day_total=total,
        day_correct=correct,
        guest_streak=guest_streak,
        img_attr=attr,
    )

@roulette_bp.post("/roulette/guess")
def submit_guess():
    r = _today_round()
    payload = request.get_json(force=True) or {}
    choice = int(payload.get("choice", -1))
    correct_index = int(payload.get("correct_index", -1))

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

    server_streak = _update_server_streak_if_logged_in()

    resp = make_response()
    today = date.today().isoformat()
    last = request.cookies.get("tr_last_play_date")
    guest_streak = int(request.cookies.get("tr_streak", "0"))
    if last != today:
        guest_streak = guest_streak + 1 if last == (date.today() - timedelta(days=1)).isoformat() else 1

    total = db.session.query(func.count(TimelineGuess.id)).filter(TimelineGuess.round_id == r.id).scalar() or 0
    correct_ct = db.session.query(func.count(TimelineGuess.id)).filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True))).scalar() or 0

    resp.set_data(
        jsonify(
            {
                "ok": True,
                "correct": is_correct,
                "totals": {"plays": total, "correct": correct_ct},
                "streak": server_streak if server_streak is not None else guest_streak,
            }
        ).get_data(as_text=True)
    )
    resp.headers["Content-Type"] = "application/json"
    resp.set_cookie("tr_streak", str(guest_streak), max_age=7 * 24 * 3600, httponly=False, samesite="Lax")
    resp.set_cookie("tr_last_play_date", today, max_age=7 * 24 * 3600, httponly=False, samesite="Lax")
    return resp

def _icon_url_or_fallback(name: str | None) -> str:
    static_dir = Path(current_app.static_folder) / "roulette" / "icons"
    for fb in ["star.svg","sparkles.svg","asterisk.svg","dot.svg","circle.svg","history.svg","compass.svg","feather.svg"]:
        if (static_dir / fb).exists():
            fallback = fb
            break
    else:
        # any svg in the folder
        fallback = next((p.name for p in static_dir.glob("*.svg")), "star.svg")

    if not name:
        return url_for("static", filename=f"roulette/icons/{fallback}")
    if (static_dir / name).exists():
        return url_for("static", filename=f"roulette/icons/{name}")
    return url_for("static", filename=f"roulette/icons/{fallback}")

# inside play_today():
cards = [
    {"orig_idx": 0, "text": r.real_title, "label": "A", "icon": _icon_url_or_fallback(r.real_icon)},
    {"orig_idx": 1, "text": r.fake1_title, "label": "B", "icon": _icon_url_or_fallback(r.fake1_icon)},
    {"orig_idx": 2, "text": r.fake2_title, "label": "C", "icon": _icon_url_or_fallback(r.fake2_icon)},
]

@roulette_bp.get("/roulette/leaderboard")
def leaderboard():
    r = _today_round()
    today_total = db.session.query(func.count(TimelineGuess.id)).filter(TimelineGuess.round_id == r.id).scalar() or 0
    today_correct = db.session.query(func.count(TimelineGuess.id)).filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True))).scalar() or 0

    seven_days_ago = date.today() - timedelta(days=6)
    rounds = db.session.query(TimelineRound.id).filter(TimelineRound.round_date >= seven_days_ago).subquery()
    week_total = db.session.query(func.count(TimelineGuess.id)).filter(TimelineGuess.round_id.in_(rounds)).scalar() or 0
    week_correct = db.session.query(func.count(TimelineGuess.id)).filter(and_(TimelineGuess.round_id.in_(rounds), TimelineGuess.is_correct.is_(True))).scalar() or 0

    return render_template(
        "roulette/leaderboard.html",
        today_total=today_total,
        today_correct=today_correct,
        week_total=week_total,
        week_correct=week_correct,
    )
