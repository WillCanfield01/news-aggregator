
# app/escape/routes.py (REWRITE)
# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
from typing import Any, Dict

from flask import Blueprint, abort, current_app, jsonify, make_response, render_template, request
from app.extensions import db

from .models import EscapeRoom, EscapeScore
from . import core  # this module is our rewritten core (generate/verify)


bp = Blueprint("escape", __name__, url_prefix="/escape")


def _ip_hash(req) -> str:
    ip = req.headers.get("X-Forwarded-For", "").split(",")[0].strip() or req.remote_addr or "0.0.0.0"
    payload = f"{ip}::{core._date_key()}".encode()
    return hashlib.sha256(payload + core._app_secret()).hexdigest()


def _ensure_room(date: str) -> EscapeRoom:
    room = EscapeRoom.query.filter_by(date=date).first()
    if room:
        return room
    # generate and persist
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
    """
    Body: { "minigame_id": "A", "transcript": {...}, "client_time_ms": 2431 }
    Returns on success: { "passed": true, "fragment": "◆", "signed_token": "..." }
    """
    data = request.get_json(force=True, silent=True) or {}
    mid = data.get("minigame_id")
    transcript = data.get("transcript") or {}
    client_time_ms = int(data.get("client_time_ms") or 0)

    if mid not in ("A", "B", "C"):
        return jsonify({"passed": False, "error": "bad minigame id"}), 400

    date = core._date_key()
    room = _ensure_room(date)

    # find cfg for mini id
    cfg = next((m for m in room.minigames_json if m.get("id") == mid), None)
    if not cfg:
        return jsonify({"passed": False, "error": "missing config"}), 400

    try:
        passed, elapsed_ms = core.verify_minigame(mid, cfg, room.server_private_json, transcript)
    except Exception as e:
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
    """
    Body: {
      "tokens": [
        {"minigame_id":"A","fragment":"◆","signed_token":"...","elapsed_ms":...},
        {"minigame_id":"B","fragment":"◎","signed_token":"...","elapsed_ms":...},
        {"minigame_id":"C","fragment":"✶","signed_token":"...","elapsed_ms":...}
      ]
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    tokens = data.get("tokens") or []
    if len(tokens) != 3:
        return jsonify({"ok": False, "error": "need 3 tokens"}), 400

    date = core._date_key()
    room = _ensure_room(date)

    # Verify tokens independently
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

    # Prevent multiple finishes per IP per day
    existing = EscapeScore.query.filter_by(date=date, ip_hash=_ip_hash(request)).first()
    if existing:
        return jsonify({"ok": True, "final_code": final_code, "total_time_ms": int(total_ms), "note": "score already recorded for today"})

    # Persist score
    score = EscapeScore(
        date=date,
        total_time_ms=int(total_ms),
        ip_hash=_ip_hash(request),
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
