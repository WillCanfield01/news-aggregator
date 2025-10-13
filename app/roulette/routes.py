from datetime import date, timedelta
import hashlib
import random
from pathlib import Path
import os
from flask import render_template, request, jsonify, abort, make_response, url_for, current_app
from sqlalchemy import func, and_, text
from urllib.parse import quote
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

def _neutral_avatar_svg() -> str:
    # simple, MIT-safe neutral circle avatar
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<circle cx="32" cy="32" r="30" fill="#f3f4f6" stroke="#e5e7eb" stroke-width="2"/>'
        '<circle cx="32" cy="26" r="10" fill="#e5e7eb"/>'
        '<rect x="14" y="40" width="36" height="14" rx="7" fill="#e5e7eb"/>'
        '</svg>'
    )

def _icon_data_uri(svg_text: str | None) -> str:
    # Always return a usable data URI
    if not svg_text or not svg_text.strip():
        svg_text = _neutral_avatar_svg()
    # Keep it utf8 (no base64) so it’s small + fast
    return "data:image/svg+xml;utf8," + svg_text.replace("#", "%23").replace("\n", "")

def _svg_data_uri(svg_text: str) -> str:
    # keep small svgs readable; encode minimally
    return f"data:image/svg+xml;utf8,{quote(svg_text)}"

def _read_icon_svg(name: str | None) -> str | None:
    static_dir = Path(current_app.static_folder) / "roulette" / "icons"

    # 1) If a specific name was provided, use it when it exists
    if name:
        p = static_dir / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8") or None
            except Exception:
                pass  # fall through to fallbacks

    # 2) Try known neutral fallbacks (first that exists)
    for fb in ["star.svg", "sparkles.svg", "asterisk.svg", "dot.svg", "circle.svg", "history.svg", "compass.svg", "feather.svg"]:
        p = static_dir / fb
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8") or None
            except Exception:
                continue

    # 3) Last resort: any svg in the folder
    any_svg = next((p for p in static_dir.glob("*.svg") if p.is_file()), None)
    if any_svg:
        try:
            return any_svg.read_text(encoding="utf-8") or None
        except Exception:
            return None
    return None

def _icon_data_uri(name: str | None) -> str | None:
    svg = _read_icon_svg(name)
    return _svg_data_uri(svg) if svg else None

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


def _icon_url_or_fallback(name: str | None) -> str:
    """
    Resolve a static URL for an icon filename under /static/roulette/icons.
    If the requested file doesn't exist, fall back to the first existing
    neutral icon; if none, pick any svg in the folder.
    """
    static_dir = Path(current_app.static_folder) / "roulette" / "icons"

    # find a fallback that actually exists
    for fb in ["star.svg", "sparkles.svg", "asterisk.svg", "dot.svg", "circle.svg", "history.svg", "compass.svg", "feather.svg"]:
        if (static_dir / fb).exists():
            fallback = fb
            break
    else:
        fallback = next((p.name for p in static_dir.glob("*.svg")), "star.svg")

    if name and (static_dir / name).exists():
        chosen = name
    else:
        chosen = fallback

    return url_for("static", filename=f"roulette/icons/{chosen}")


# -------- Pages --------
@roulette_bp.get("/roulette")
def play_today():
    r = _today_round()

    # Build cards here (after we have r), using safe icon URLs
    cards = [
        {"orig_idx": 0, "text": r.real_title, "label": "A",
        "icon": _icon_data_uri(_read_icon_svg(r.real_icon))},
        {"orig_idx": 1, "text": r.fake1_title, "label": "B",
        "icon": _icon_data_uri(_read_icon_svg(r.fake1_icon))},
        {"orig_idx": 2, "text": r.fake2_title, "label": "C",
        "icon": _icon_data_uri(_read_icon_svg(r.fake2_icon))},
    ]
    random.shuffle(cards)
    correct_shuffled_idx = next(i for i, c in enumerate(cards) if c["orig_idx"] == 0)

    total = db.session.query(func.count(TimelineGuess.id)).filter(TimelineGuess.round_id == r.id).scalar() or 0
    correct = db.session.query(func.count(TimelineGuess.id)).filter(and_(TimelineGuess.round_id == r.id, TimelineGuess.is_correct.is_(True))).scalar() or 0

    guest_streak = request.cookies.get("tr_streak")

    # Optional: keep attribution if you still show it somewhere
    attr = getattr(r, "real_img_attr", None) or getattr(r, "fake1_img_attr", None) or getattr(r, "fake2_img_attr", None)

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

@roulette_bp.route("/roulette/admin/regen", methods=["GET", "POST"])
def roulette_admin_regen():
    """
    Regenerate today's Timeline Roulette round.
    Security: requires token via header X-Admin-Token or query ?token=...
    Query option: force=1  (clears today's guesses + round first)
    """
    import os
    from sqlalchemy import text
    from datetime import date
    from flask import request, jsonify
    from app.extensions import db
    from app.roulette.models import TimelineRound
    from app.scripts.generate_timeline_round import ensure_today_round

    token = request.headers.get("X-Admin-Token") or request.args.get("token")
    expected = os.getenv("ROULETTE_ADMIN_TOKEN")
    if not expected or token != expected:
        return jsonify({"ok": False, "error": "forbidden"}), 403

    force = str(request.args.get("force", "0")).lower() in ("1", "true", "yes")
    if force:
        db.session.execute(text("""
            DELETE FROM timeline_guesses
            WHERE round_id IN (SELECT id FROM timeline_rounds WHERE round_date = CURRENT_DATE);
        """))
        db.session.execute(text("""
            DELETE FROM timeline_rounds
            WHERE round_date = CURRENT_DATE;
        """))
        db.session.commit()

    ensure_today_round()

    r = TimelineRound.query.filter_by(round_date=date.today()).first()
    if not r:
        return jsonify({"ok": False, "error": "round_not_created"}), 500

    return jsonify({
        "ok": True,
        "round_date": str(r.round_date),
        "real_title": r.real_title,
        "fake1_title": r.fake1_title,
        "fake2_title": r.fake2_title,
        "correct_index": r.correct_index
    })

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
