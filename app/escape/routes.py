
# app/escape/routes.py (FULL with legacy init_routes compatibility)
# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any, Dict

from flask import Blueprint, jsonify, render_template, request
from app.extensions import db

from .models import EscapeRoom, EscapeScore
from . import core
import os

bp = Blueprint("escape", __name__, url_prefix="/escape")


def _ip_hash(req) -> str:
    ip = req.headers.get("X-Forwarded-For", "").split(",")[0].strip() or req.remote_addr or "0.0.0.0"
    payload = f"{ip}::{core._date_key()}".encode()
    return hashlib.sha256(payload + core._app_secret()).hexdigest()


def _ensure_room(date: str) -> EscapeRoom:
    room = EscapeRoom.query.filter_by(date=date).first()
    if room:
        return room
    dr = core.generate_room(dt.datetime.strptime(date, "%Y-%m-%d").date())
    room = EscapeRoom(
        date=dr.date,
        theme=dr.theme,
        minigames_json=dr.minigames,
        server_private_json=dr.server_private,
    )
    db.session.add(room)
    db.session.commit()
    return room


@bp.get("/api/today")
def api_today():
    date = request.args.get("date") or core._date_key()
    room = _ensure_room(date)
    return jsonify(core.room_public_payload(core.DailyRoom(
        date=room.date,
        theme=room.theme,
        minigames=room.minigames_json,
        server_private=room.server_private_json,
    )))


@bp.post("/api/submit")
def api_submit():
    data = request.get_json(force=True, silent=True) or {}
    mid = data.get("minigame_id")
    transcript = data.get("transcript") or {}
    client_time_ms = int(data.get("client_time_ms") or 0)

    if mid not in ("A", "B", "C"):
        return jsonify({"passed": False, "error": "bad minigame id"}), 400

    date = core._date_key()
    room = _ensure_room(date)

    cfg = next((m for m in room.minigames_json if m.get("id") == mid), None)
    if not cfg:
        return jsonify({"passed": False, "error": "missing config"}), 400

    try:
        passed, elapsed_ms = core.verify_minigame(mid, cfg, room.server_private_json, transcript)
    except Exception:
        return jsonify({"passed": False, "error": "verify-failed"}), 400

    if not passed:
        return jsonify({"passed": False}), 200

    fragment = core.fragment_for(mid)
    envelope = {"date": date, "minigame_id": mid, "fragment": fragment, "elapsed_ms": int(elapsed_ms)}
    signed = core._hmac_sign(envelope)

    return jsonify({
        "passed": True,
        "fragment": fragment,
        "signed_token": signed,
        "elapsed_ms": int(elapsed_ms),
    })


@bp.post("/api/finish")
def api_finish():
    data = request.get_json(force=True, silent=True) or {}
    tokens = data.get("tokens") or []
    if len(tokens) != 3:
        return jsonify({"ok": False, "error": "need 3 tokens"}), 400

    date = core._date_key()
    room = _ensure_room(date)

    fragments = []
    total_ms = 0
    for t in tokens:
        env = {
            "date": date,
            "minigame_id": t.get("minigame_id"),
            "fragment": t.get("fragment"),
            "elapsed_ms": int(t.get("elapsed_ms") or 0),
        }
        sig = t.get("signed_token", "")
        if not core._hmac_verify(env, sig):
            return jsonify({"ok": False, "error": "bad token"}), 400
        fragments.append(env["fragment"])
        total_ms += env["elapsed_ms"]

    final_code = core.assemble_final_code(fragments)

    # One score per IP/date
    this_ip = _ip_hash(request)
    existing = EscapeScore.query.filter_by(date=date, ip_hash=this_ip).first()
    if existing:
        return jsonify({"ok": True, "final_code": final_code, "total_time_ms": int(total_ms), "note": "score already recorded for today"})

    score = EscapeScore(
        date=date,
        total_time_ms=int(total_ms),
        ip_hash=this_ip,
    )
    db.session.add(score)
    db.session.commit()

    return jsonify({"ok": True, "final_code": final_code, "total_time_ms": int(total_ms)})


@bp.get("/play")
def play_today():
    return render_template("escape/play.html")


@bp.get("/leaderboard")
def leaderboard_view():
    date = request.args.get("date") or core._date_key()
    rows = EscapeScore.top_for_day(date=date, limit=100)
    return render_template("escape/leaderboard.html", date=date, rows=rows)

@bp.route("/api/admin/regen", methods=["GET", "POST"])
def api_admin_regen():
    """
    Admin-only: regenerate (or rotate) a room for a given date.
    Params:
      - token: must match ESCAPE_ADMIN_TOKEN env var (required)
      - date: YYYY-MM-DD (defaults to today)
      - force: true/false (kept for compat; not required)
      - rotate or salt: any string; if provided, creates a different variant for the same date
    """
    expected = os.getenv("ESCAPE_ADMIN_TOKEN")
    token = request.args.get("token") or (request.get_json(silent=True) or {}).get("token")
    if not expected or token != expected:
        return jsonify({"ok": False, "error": "forbidden"}), 403

    date = request.args.get("date") or (request.get_json(silent=True) or {}).get("date") or core._date_key()
    salt = (
        request.args.get("rotate") or request.args.get("salt")
        or (request.get_json(silent=True) or {}).get("rotate")
        or (request.get_json(silent=True) or {}).get("salt")
    )

    try:
        d = dt.datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"ok": False, "error": "bad date"}), 400

    # Generate (optionally rotated with salt) and upsert
    try:
        room = core.generate_room(d, salt=str(salt))  # salt may be None → normal daily
    except TypeError:
        # If your core doesn't have salt yet, fall back to normal generate
        room = core.generate_room(d)

    existing = EscapeRoom.query.filter_by(date=room.date).first()
    if existing:
        existing.theme = room.theme
        existing.minigames_json = room.minigames
        existing.server_private_json = room.server_private
        db.session.add(existing)
    else:
        db.session.add(EscapeRoom(
            date=room.date, theme=room.theme,
            minigames_json=room.minigames, server_private_json=room.server_private
        ))
    db.session.commit()

    return jsonify({"ok": True, "date": room.date, "theme": room.theme,
                    "minis": [m.get("type") for m in room.minigames]})


# ───────────────────── Legacy compatibility: init_routes(bp) ─────────────────────
def init_routes(bp_external=None):
    """
    Some existing setups call `from .routes import init_routes` and pass in a Blueprint
    created elsewhere (e.g., in app/escape/__init__.py). We attach our view funcs to it.
    If no blueprint is provided, return the module-level `bp` so callers can register it.
    """
    if bp_external is None:
        return bp
    bp_external.add_url_rule("/api/today", view_func=api_today, methods=["GET"])
    bp_external.add_url_rule("/api/submit", view_func=api_submit, methods=["POST"])
    bp_external.add_url_rule("/api/finish", view_func=api_finish, methods=["POST"])
    bp_external.add_url_rule("/play", view_func=play_today, methods=["GET"])
    bp_external.add_url_rule("/leaderboard", view_func=leaderboard_view, methods=["GET"])
    return bp_external
