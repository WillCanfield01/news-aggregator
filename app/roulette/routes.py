# app/roulette/routes.py
from __future__ import annotations

import hashlib
import os
import random
import uuid
import json
import requests
import threading
import time
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
    session,
)
from sqlalchemy import and_, func

from app.extensions import db
from app.plus import get_plus_checkout_url
from app.subscriptions import current_user_is_plus, plus_required

# Optional: support without flask-login present
try:
    from flask_login import current_user  # type: ignore
except Exception:  # pragma: no cover
    class _Anon:
        is_authenticated = False
        id = None
    current_user = _Anon()  # type: ignore

from . import roulette_bp
from .models import TimelineGuess, TimelineRound, TimelineStreak, RouletteSession, RouletteRegenJob
from .admin_jobs import enqueue_job, job_status_payload, get_running_job, start_worker

# reuse generators
from app.scripts.generate_timeline_round import (
    OPENAI_API_KEY,
    OAI_MODEL,
    OAI_URL,
    QUOTE_LIBRARY,
    _http_get_json,
    _headline_from_text,
    _blurb_from_text,
    pick_image_for_choice,
    _validate_image_url,
    _guess_event_type,
    _category_for_text,
    _guess_location,
    _guess_entity,
    _infer_domain,
    _extract_year_hint,
    ensure_today_round,
)


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

# -------------------------------------------------------------------------
# Session helpers (multi-step game)
# -------------------------------------------------------------------------
_SESSION_COOKIE = "tr_session_id"
_ADMIN_TOKEN = os.getenv("ADMIN_REGEN_TOKEN", "")
_runner_started = False
_runner_lock = threading.Lock()
_process_lock = threading.Lock()
_ROULETTE_PLAY_KEY = "rr_roulette_played"


def free_roulette_available(today: date | None = None) -> bool:
    """
    Return True if the user/session has not used today's free roulette play.
    """
    today_label = (today or _local_today()).isoformat()
    try:
        played = session.get(_ROULETTE_PLAY_KEY)
    except Exception:
        played = None
    return played != today_label


def mark_free_roulette_used(today: date | None = None):
    """
    Mark today's free roulette play as consumed for this session/user.
    """
    try:
        session[_ROULETTE_PLAY_KEY] = (today or _local_today()).isoformat()
    except Exception:
        pass


def _log_gate_decision(session_id: str | None, decision: str, plus_user: bool, free_available: bool, blocked_reason: str | None = None):
    """
    Minimal logging to understand roulette gating decisions.
    """
    try:
        current_app.logger.info(
            "roulette_session_gate user=%s session=%s plus=%s free_available=%s decision=%s blocked_reason=%s",
            getattr(current_user, "id", None) or "anon",
            session_id or _get_session_id() or "none",
            plus_user,
            free_available,
            decision,
            blocked_reason or "",
        )
    except Exception:
        pass

def _set_session_cookie(resp, sid: str):
    resp.set_cookie(
        _SESSION_COOKIE,
        sid,
        httponly=True,
        secure=True,
        samesite="Lax",
        max_age=60 * 60 * 12,
    )

def _get_session_id() -> str | None:
    return request.cookies.get(_SESSION_COOKIE)

def _new_session(today: date, user_id: int | None = None) -> RouletteSession:
    sid = RouletteSession.new_id()
    sess = RouletteSession(
        id=sid,
        round_date=today,
        current_step=1,
        score=0,
        user_id=user_id,
    )
    db.session.add(sess)
    db.session.commit()
    return sess

def _clear_session_cookie(resp):
    resp.delete_cookie(_SESSION_COOKIE)


def _admin_auth_or_abort():
    token = request.headers.get("X-Admin-Token", "")
    if not _ADMIN_TOKEN or token != _ADMIN_TOKEN:
        abort(401)


# -------------------------------------------------------------------------
# Regen background runner
# -------------------------------------------------------------------------
def run_regen_job(job_id: int):
    # kept for backward compatibility; new runner lives in admin_jobs.py
    from .admin_jobs import _run_job as _r
    _r(job_id)


def _regen_worker(app):
    # compatibility shim; real worker runs in admin_jobs.start_worker
    from .admin_jobs import _worker as _w
    _w(app)


def start_regen_worker(app):
    from .admin_jobs import start_worker as _s
    _s(app)


@roulette_bp.post("/roulette/session/reset")
@plus_required
def reset_session():
    sid = _get_session_id()
    if sid:
        sess = RouletteSession.query.filter_by(id=sid).first()
        if sess:
            db.session.delete(sess)
            db.session.commit()
    resp = make_response(jsonify({"ok": True}))
    _clear_session_cookie(resp)
    return resp

# -------------------------------------------------------------------------
# Round builders
# -------------------------------------------------------------------------
def _shuffle_cards(cards: list[dict]) -> tuple[list[dict], int]:
    random.shuffle(cards)
    correct_shuffled_idx = next(i for i, c in enumerate(cards) if c.get("orig_idx") == 0)
    return cards, correct_shuffled_idx


def _choice_cards(today_round: TimelineRound, include_blurb: bool = False) -> list[dict]:
    """
    Build base card payloads from today's round with headline + blurb + image.
    """
    def _one(idx: int, text: str, img: str | None, icon_name: str | None) -> dict:
        blurb = _blurb_from_text(text, 12, 22) if include_blurb else ""
        return {
            "orig_idx": idx,
            "title": _headline_from_text(text, 6, 14),
            "blurb": blurb,
            "image_url": img or _icon_url_or_fallback(icon_name),
            "raw_text": text,
        }

    return [
        _one(0, today_round.real_title, getattr(today_round, "real_img_url", None), getattr(today_round, "real_icon", None)),
        _one(1, today_round.fake1_title, getattr(today_round, "fake1_img_url", None), getattr(today_round, "fake1_icon", None)),
        _one(2, today_round.fake2_title, getattr(today_round, "fake2_img_url", None), getattr(today_round, "fake2_icon", None)),
    ]

def _build_text_round(today_round: TimelineRound) -> dict:
    cards, correct_idx = _shuffle_cards(_choice_cards(today_round, include_blurb=False))
    date_label = today_round.round_date.strftime("%B %d")
    return {
        "type": "text",
        "puzzle_type": "HEADLINE",
        "prompt": "Three headlines. One is real.",
        "subhead": f"Pick the real headline. Based on an event from {date_label}.",
        "date_label": date_label,
        "cards": cards,
        "correct_index": correct_idx,
        "source_url": today_round.real_source_url,
    }

def _build_image_round(today_round: TimelineRound) -> dict:
    date_label = today_round.round_date.strftime("%B %d")
    # build fresh image-only cards to avoid reusing headline puzzle images
    def _choice_meta(text: str) -> dict:
        ev_type, ev_cat = _guess_event_type(text)
        domain_guess = _infer_domain(text)
        category = ev_cat or _category_for_text(text, domain_guess)
        return {
            "text": text,
            "domain": domain_guess,
            "category": category,
            "event_type": ev_type or category,
            "location": _guess_location(text),
            "entity": _guess_entity(text),
            "year": _extract_year_hint(text),
        }

    used_urls: set[str] = set(
        u for u in [
            _validate_image_url(getattr(today_round, "real_img_url", None)),
            _validate_image_url(getattr(today_round, "fake1_img_url", None)),
            _validate_image_url(getattr(today_round, "fake2_img_url", None)),
        ] if u
    )
    seed_base = int(datetime.now(_TZ).strftime("%Y%m%d"))
    rng = random.Random(seed_base + 101)
    real_meta = _choice_meta(today_round.real_title)
    fake1_meta = _choice_meta(today_round.fake1_title)
    fake2_meta = _choice_meta(today_round.fake2_title)
    real_img, _, _ = pick_image_for_choice(real_meta, rng, used_urls, None, mode="real")
    f1_img, _, _ = pick_image_for_choice(fake1_meta, rng, used_urls, None, mode="decoy", seed=seed_base + 211)
    f2_img, _, _ = pick_image_for_choice(fake2_meta, rng, used_urls, None, mode="decoy", seed=seed_base + 223)

    cards, correct_idx = _shuffle_cards([
        {"orig_idx": 0, "title": "", "blurb": "", "image_url": real_img, "raw_text": today_round.real_title},
        {"orig_idx": 1, "title": "", "blurb": "", "image_url": f1_img, "raw_text": today_round.fake1_title},
        {"orig_idx": 2, "title": "", "blurb": "", "image_url": f2_img, "raw_text": today_round.fake2_title},
    ])
    return {
        "type": "image",
        "puzzle_type": "PHOTO",
        "prompt": "Three photos. One is real.",
        "subhead": f"Pick the photo that matches a real event. Based on {date_label}.",
        "date_label": date_label,
        "cards": cards,
        "correct_index": correct_idx,
        "source_url": today_round.real_source_url,
    }

def _fetch_today_quote() -> tuple[str, str]:
    """
    Return (quote_text, attribution) for today via Wikiquote; fallback to empty strings.
    """
    try:
        j = _http_get_json(
            "https://en.wikiquote.org/w/api.php",
            params={
                "action": "parse",
                "page": "Template:Today/Quotes",
                "prop": "wikitext",
                "format": "json",
            },
        )
        wikitext = j.get("parse", {}).get("wikitext", {}).get("*", "")
        # naive parse: split bullet lines
        lines = [ln.strip("* ").strip() for ln in wikitext.splitlines() if ln.strip().startswith("*")]
        if lines:
            first = lines[0]
            if " - " in first:
                quote, author = first.split(" - ", 1)
                return quote.strip(), author.strip()
            return first, ""
    except Exception:
        pass
    return "", ""

def _quote_tokens(q: str) -> set[str]:
    return {w.lower().strip(".,;:!?\"'") for w in (q or "").split() if w}

def _quote_too_similar(a: str, b: str) -> bool:
    ta, tb = _quote_tokens(a), _quote_tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb) or 1
    return inter / union > 0.55

def _clean_quote_text(text: str) -> str:
    t = (text or "").strip()
    t = t.strip("\"'“”‘’ ")
    return t


def _quote_images() -> list[str]:
    static_dir = Path(current_app.static_folder) / "roulette" / "fallbacks"
    preferred = ["culture-01.svg", "general-01.svg", "culture-02.svg", "general-02.svg"]
    urls: list[str] = []
    for name in preferred:
        if (static_dir / name).exists():
            urls.append(url_for("static", filename=f"roulette/fallbacks/{name}"))
    if not urls:
        urls.append(_icon_url_or_fallback("star.svg"))
    while len(urls) < 3:
        urls.append(urls[-1])
    return urls[:3]


def _quote_pool() -> list[dict]:
    return QUOTE_LIBRARY


def _pick_real_quote(seed: int, avoid_authors: set[str]) -> tuple[str, str]:
    rng = random.Random(seed)
    pool = list(_quote_pool())
    rng.shuffle(pool)
    for item in pool:
        author = (item.get("author") or "").strip()
        if author and author not in avoid_authors:
            return (item.get("text") or "").strip(), author
    fallback = pool[0] if pool else {"text": "A recorded quote.", "author": "Unknown"}
    return fallback.get("text", ""), fallback.get("author", "")


def _fabricate_quote(avoid_authors: set[str], rng: random.Random) -> tuple[str, str]:
    names = [
        "Clara Mendes", "Julian Hart", "Priya Raman", "Seth Alvarez", "Mara Quinn",
        "Devon Carlisle", "Noor Hassan", "Inez Porter", "Caleb Brooks", "Lena Duarte",
        "Riley Cobb", "Anika Shah", "Mason Pike", "Talia Rhodes", "Jonah Reeves",
        "Kai Morgan", "Vivian Cross", "Leo Navarro", "Isla Becker", "Tomas Reed",
        "Felix Rowe", "Daria Voss", "Kendall Price", "Mateo Serrano",
    ]
    openers = [
        "I carry the lesson that",
        "What stays with me is that",
        "I learned long ago that",
        "We move forward when",
        "It is never wasted when",
        "Progress starts the moment",
        "I hold to the idea that",
        "We earn our momentum when",
    ]
    middles = [
        "steady effort meets honest intent",
        "quiet practice builds real strength",
        "curiosity outruns the fear of failing",
        "discipline and hope travel together",
        "we choose to keep showing up",
        "careful work honors the craft",
        "we listen before we leap",
        "we take the next small step",
    ]
    endings = [
        "and that is enough to change a day.",
        "and that turns work into craft.",
        "and that is the root of good work.",
        "and that keeps the door open.",
        "and that is where courage lives.",
        "and that is how a team begins.",
        "and that is how we earn trust.",
        "and that is how progress holds.",
    ]
    rng.shuffle(names)
    author = next((n for n in names if n not in avoid_authors), names[0])
    text = f"{rng.choice(openers)} {rng.choice(middles)} {rng.choice(endings)}"
    return text.strip(), author

def _build_quote_round(today_round: TimelineRound) -> dict:
    seed = int(datetime.now(_TZ).strftime("%Y%m%d"))
    real_quote, real_author = _pick_real_quote(seed, set())
    avoid = {real_author}
    fake1, fake1_author = _fabricate_quote(avoid, random.Random(seed + 11))
    avoid.add(fake1_author)
    fake2, fake2_author = _fabricate_quote(avoid, random.Random(seed + 19))

    images = _quote_images()
    cards, correct_idx = _shuffle_cards([
        {
            "orig_idx": 0,
            "title": real_quote or "A historical quote recorded for this date.",
            "quote": real_quote or "A historical quote recorded for this date.",
            "author": real_author,
            "blurb": real_author,
            "image_url": images[0],
            "raw_text": real_quote or "",
        },
        {
            "orig_idx": 1,
            "title": fake1,
            "quote": fake1,
            "author": fake1_author,
            "blurb": fake1_author,
            "image_url": images[1],
            "raw_text": fake1,
        },
        {
            "orig_idx": 2,
            "title": fake2,
            "quote": fake2,
            "author": fake2_author,
            "blurb": fake2_author,
            "image_url": images[2],
            "raw_text": fake2,
        },
    ])
    return {
        "type": "quote",
        "puzzle_type": "QUOTE",
        "prompt": "One quote is real.",
        "subhead": "Two are decoys. Pick the real line.",
        "date_label": today_round.round_date.strftime("%B %d"),
        "cards": cards,
        "correct_index": correct_idx,
        "source_url": today_round.real_source_url,
    }


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
@roulette_bp.get("/roulette")
def play_today():
    r = _today_round_or_404()

    # Build shuffled card set; remember where the real one lands (index 0 == real)
    cards, correct_shuffled_idx = _shuffle_cards(_choice_cards(r))

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


@roulette_bp.get("/roulette/admin")
def roulette_admin_page():
    if not _ADMIN_TOKEN:
        abort(404)
    return render_template("roulette/admin.html", admin_enabled=bool(_ADMIN_TOKEN))


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
# Multi-round session API (3-step: text, image, quote)
# -----------------------------------------------------------------------------
def _get_or_create_session():
    today = _local_today()
    sid = _get_session_id()
    sess = None
    if sid:
        sess = RouletteSession.query.filter_by(id=sid).first()
        if sess and (sess.round_date != today or sess.current_step > 3):
            db.session.delete(sess)
            db.session.commit()
            sess = None
    plus_user = current_user_is_plus()
    free_available = free_roulette_available(today)
    decision = "resume" if sess else None
    if not sess and plus_user:
        sess = _new_session(
            today,
            getattr(current_user, "id", None) if getattr(current_user, "is_authenticated", False) else None,
        )
        decision = "plus_new"
    elif not sess and free_available:
        sess = _new_session(
            today,
            getattr(current_user, "id", None) if getattr(current_user, "is_authenticated", False) else None,
        )
        decision = "free_new"
        mark_free_roulette_used(today)
    elif not sess:
        decision = "blocked"

    # Ensure we record the free play once a session exists for today
    if sess and not plus_user and free_available:
        mark_free_roulette_used(today)

    return sess, {
        "plus_user": plus_user,
        "free_available": free_available,
        "decision": decision or ("resume" if sess else "blocked"),
    }

def _payload_for_step(sess: RouletteSession, today_round: TimelineRound) -> dict:
    if sess.current_step == 1:
        if not sess.text_payload:
            sess.text_payload = _build_text_round(today_round)
            db.session.commit()
        return sess.text_payload
    if sess.current_step == 2:
        if not sess.image_payload:
            sess.image_payload = _build_image_round(today_round)
            db.session.commit()
        return sess.image_payload
    if sess.current_step == 3:
        if not sess.quote_payload:
            sess.quote_payload = _build_quote_round(today_round)
            db.session.commit()
        return sess.quote_payload
    return {}


def _plus_required_response(status_code: int = 402):
    payload = {
        "ok": False,
        "reason": "plus_required",
        "error": "plus_required",
        "title": "Want to keep playing?",
        "message": "Plus unlocks unlimited plays, streak protection, and past days.",
        "next_action": "upgrade_plus",
        "checkout_url": get_plus_checkout_url(),
    }
    return jsonify(payload), status_code


@roulette_bp.get("/roulette/session")
def get_session():
    today_round = _today_round_or_404()
    sess, gate = _get_or_create_session()
    plus_user = gate.get("plus_user", False)
    free_available = gate.get("free_available", False)
    decision = gate.get("decision", "unknown")
    if sess is None:
        _log_gate_decision(None, decision, plus_user, free_available, "plus_required")
        return _plus_required_response(200)
    payload = _payload_for_step(sess, today_round)
    _log_gate_decision(sess.id, decision, plus_user, free_available, None)
    resp = make_response(jsonify({
        "ok": True,
        "session_id": sess.id,
        "step": sess.current_step,
        "score": sess.score,
        "payload": payload,
        "guesses": {
            "text": sess.text_guess,
            "image": sess.image_guess,
            "quote": sess.quote_guess,
        },
    }))
    _set_session_cookie(resp, sess.id)
    return resp

@roulette_bp.post("/roulette/session/guess")
def session_guess():
    today_round = _today_round_or_404()
    sess, gate = _get_or_create_session()
    if sess is None:
        _log_gate_decision(None, gate.get("decision", "blocked"), gate.get("plus_user", False), gate.get("free_available", False), "plus_required")
        return _plus_required_response(402)
    if sess.current_step > 3:
        return jsonify({"ok": False, "error": "session_finished"}), 400
    payload = request.get_json(force=True) or {}
    try:
        choice = int(payload.get("choice", -1))
    except Exception:
        return jsonify({"ok": False, "error": "bad_input"}), 400
    cur_payload = _payload_for_step(sess, today_round)
    correct = cur_payload.get("correct_index", -1)
    if choice not in (0, 1, 2) or correct not in (0, 1, 2):
        return jsonify({"ok": False, "error": "bad_input"}), 400

    is_correct = (choice == correct)
    if sess.current_step == 1:
        sess.text_guess = choice
    elif sess.current_step == 2:
        sess.image_guess = choice
    elif sess.current_step == 3:
        sess.quote_guess = choice
    if is_correct:
        sess.score += 1
    db.session.commit()
    try:
        mark_free_roulette_used(_local_today())
    except Exception:
        pass

    return jsonify({"ok": True, "correct_index": correct, "is_correct": is_correct, "score": sess.score})

@roulette_bp.post("/roulette/session/next")
def session_next():
    today_round = _today_round_or_404()
    sess, gate = _get_or_create_session()
    if sess is None:
        _log_gate_decision(None, gate.get("decision", "blocked"), gate.get("plus_user", False), gate.get("free_available", False), "plus_required")
        return _plus_required_response(402)
    if sess.current_step < 3:
        sess.current_step += 1
        db.session.commit()
        payload = _payload_for_step(sess, today_round)
        return jsonify({"ok": True, "step": sess.current_step, "payload": payload, "score": sess.score})
    recap = {
        "ok": True,
        "step": sess.current_step,
        "score": min(sess.score, 3),
        "rounds": {
            "text": {"payload": sess.text_payload, "guess": sess.text_guess},
            "image": {"payload": sess.image_payload, "guess": sess.image_guess},
            "quote": {"payload": sess.quote_payload, "guess": sess.quote_guess},
        },
    }
    # mark finished to prevent further guessing on this session
    sess.current_step = 4
    db.session.commit()
    try:
        mark_free_roulette_used(_local_today())
    except Exception:
        pass
    return jsonify(recap)


# -----------------------------------------------------------------------------
# Admin: hit this to force-generate today's round
# GET /roulette/admin/regen?token=...&force=1
# -----------------------------------------------------------------------------
# Admin regen job enqueuer
@roulette_bp.route("/roulette/admin/regen", methods=["GET", "POST"])
def admin_regen():
    _admin_auth_or_abort()
    running = get_running_job()
    if running:
        return (
            jsonify(
                {
                    "error": "already_running",
                    "job_id": running.id,
                    "status_url": url_for("roulette.regen_status", job_id=running.id, _external=True),
                }
            ),
            409,
        )
    force_raw = request.args.get("force", "0")
    try:
        force = int(force_raw)
    except Exception:
        force = 0
    if force not in (0, 1, 2):
        force = 0

    job = enqueue_job(force, request.headers.get("X-Forwarded-For", request.remote_addr))
    return (
        jsonify(
            {
                "job_id": job.id,
                "status": job.status,
                "status_url": url_for("roulette.regen_status", job_id=job.id, _external=True),
            }
        ),
        202,
    )


@roulette_bp.get("/roulette/admin/regen/status/<int:job_id>")
def regen_status(job_id: int):
    _admin_auth_or_abort()
    job = RouletteRegenJob.query.filter_by(id=job_id).first_or_404()
    return jsonify(job_status_payload(job))


# Manual verification checklist:
# - Load /roulette as an anonymous user and confirm the daily session starts without a 402 or crash.
# - Finish the round as a free user; attempting to restart should render the Plus card inline.
# - Mark a user as Plus (active entitlement) and confirm restart creates a fresh session smoothly.
# - Trigger a blocked replay (402 from /roulette/session/reset) and confirm the CTA points to the configured checkout URL.
