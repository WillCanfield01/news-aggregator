# app/roulette/routes.py
from __future__ import annotations

import hashlib
import os
import random
import uuid
import json
import requests
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
from .models import TimelineGuess, TimelineRound, TimelineStreak, RouletteSession

# reuse generators
from app.scripts.generate_timeline_round import (
    OPENAI_API_KEY,
    OAI_MODEL,
    OAI_URL,
    UNSPLASH_ACCESS_KEY,
    _openai_image,
    _http_get_json,
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


@roulette_bp.post("/roulette/session/reset")
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
def _shuffle_cards(real_text: str, fake1: str, fake2: str, real_img: str | None, fake1_img: str | None, fake2_img: str | None) -> tuple[list[dict], int]:
    cards = [
        {"orig_idx": 0, "text": real_text, "img": real_img},
        {"orig_idx": 1, "text": fake1, "img": fake1_img},
        {"orig_idx": 2, "text": fake2, "img": fake2_img},
    ]
    random.shuffle(cards)
    correct_shuffled_idx = next(i for i, c in enumerate(cards) if c["orig_idx"] == 0)
    return cards, correct_shuffled_idx

def _build_text_round(today_round: TimelineRound) -> dict:
    cards, correct_idx = _shuffle_cards(
        today_round.real_title,
        today_round.fake1_title,
        today_round.fake2_title,
        getattr(today_round, "real_img_url", None),
        getattr(today_round, "fake1_img_url", None),
        getattr(today_round, "fake2_img_url", None),
    )
    return {
        "type": "text",
        "prompt": f"Two are fakes. One really happened on {today_round.round_date.strftime('%B %d')}. Pick the real one.",
        "cards": cards,
        "correct_index": correct_idx,
        "source_url": today_round.real_source_url,
    }

def _build_image_round(today_round: TimelineRound) -> dict:
    # Real image strictly from Unsplash (if not already present)
    real_img = getattr(today_round, "real_img_url", None)
    if not real_img and UNSPLASH_ACCESS_KEY:
        try:
            j = _http_get_json(
                "https://api.unsplash.com/search/photos",
                params={"query": today_round.real_title, "orientation": "landscape", "per_page": 3},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            )
            if j.get("results"):
                real_img = j["results"][0]["urls"]["small"]
        except Exception:
            real_img = None
    # Fallback real image via OpenAI if Unsplash missing/failed
    if not real_img and OPENAI_API_KEY:
        real_img = _openai_image(f"photograph, realistic, news photo about: {today_round.real_title}")

    # AI decoys: generate images with OpenAI; minimal/blank captions
    ai_cards = []
    topics = ["city skyline at night", "festival crowd", "historical archive photo", "sports arena", "museum interior", "harbor at sunset"]
    for i in range(2):
        prompt_topic = random.choice(topics)
        img_url = None
        if OPENAI_API_KEY:
            img_url = _openai_image(f"photograph, realistic, {prompt_topic}, unrelated to: {today_round.real_title}")
        if not img_url and UNSPLASH_ACCESS_KEY:
            try:
                j = _http_get_json(
                    "https://api.unsplash.com/search/photos",
                    params={"query": prompt_topic, "orientation": "landscape", "per_page": 1},
                    headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                )
                if j.get("results"):
                    img_url = j["results"][0]["urls"]["small"]
            except Exception:
                img_url = None
        ai_cards.append({"text": "", "img": img_url or ""})

    cards, correct_idx = _shuffle_cards(
        "",  # no caption for real
        ai_cards[0]["text"],
        ai_cards[1]["text"],
        real_img,
        ai_cards[0]["img"],
        ai_cards[1]["img"],
    )
    return {
        "type": "image",
        "prompt": "One photo is real. Two are AI. Pick the real one.",
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

def _build_quote_round(today_round: TimelineRound) -> dict:
    real_quote, real_attr = _fetch_today_quote()
    if not real_quote and OPENAI_API_KEY:
        try:
            prompt = (
                "Provide one verifiable historical quote that occurred on today's date. "
                "Return one sentence containing the quote and the speaker/attribution. No sources or commentary."
            )
            r = requests.post(
                OAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": OAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                },
                timeout=12,
            )
            r.raise_for_status()
            real_quote = (r.json()["choices"][0]["message"]["content"] or "").strip()
            real_attr = ""
        except Exception:
            real_quote = ""
            real_attr = ""

    def _ai_quote() -> str:
        if OPENAI_API_KEY:
            try:
                r = requests.post(
                    OAI_URL,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": OAI_MODEL,
                        "messages": [
                            {"role": "system", "content": "Invent a concise, believable quote with attribution, 12-18 words, one sentence, neutral tone."},
                            {"role": "user", "content": f"Do NOT mention the real event. Make it feel like historical or cultural commentary. Real event: {today_round.real_title}."},
                        ],
                        "temperature": 0.9,
                    },
                    timeout=12,
                )
                r.raise_for_status()
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                return ""
        return ""

    fake1 = _ai_quote() or "A noted figure reflects on change in a short remark."
    fake2 = _ai_quote() or "An artist speaks on creativity in a brief line."

    cards, correct_idx = _shuffle_cards(
        real_quote or "A historical quote recorded for this date.",
        fake1,
        fake2,
        None,
        None,
        None,
    )
    return {
        "type": "quote",
        "prompt": "One quote is real. Two are AI. Pick the real one.",
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
    if not sess:
        sess = _new_session(today, getattr(current_user, "id", None) if getattr(current_user, "is_authenticated", False) else None)
    return sess

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

@roulette_bp.get("/roulette/session")
def get_session():
    today_round = _today_round_or_404()
    sess = _get_or_create_session()
    payload = _payload_for_step(sess, today_round)
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
    sess = _get_or_create_session()
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

    return jsonify({"ok": True, "correct_index": correct, "is_correct": is_correct, "score": sess.score})

@roulette_bp.post("/roulette/session/next")
def session_next():
    today_round = _today_round_or_404()
    sess = _get_or_create_session()
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
    return jsonify(recap)


# -----------------------------------------------------------------------------
# Admin: hit this to force-generate today's round
# GET /roulette/admin/regen?token=...&force=1
# -----------------------------------------------------------------------------
@roulette_bp.route("/roulette/admin/regen", methods=["GET", "POST"])
def admin_regen():
    expected = os.getenv("ROULETTE_ADMIN_TOKEN", "")
    token = request.args.get("token") or request.headers.get("X-Admin-Token", "")
    if not expected or token != expected:
        # Hide existence of this route
        abort(404)

    # Parse force as an int: 0=no-op if exists, 1=update in place, 2=wipe guesses then update
    force_raw = request.args.get("force", "1")
    try:
        force = int(force_raw)
    except Exception:
        force = 1
    if force not in (0, 1, 2):
        force = 1

    # Late import to avoid circulars
    from app.scripts.generate_timeline_round import ensure_today_round

    ok = bool(ensure_today_round(force=force))
    resp = jsonify({
        "ok": ok,
        "date": _local_today().isoformat(),
        "tz": _TZ_NAME,
        "force": force,
    })
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp, (200 if ok else 500)
