# app/escape/routes.py
# -*- coding: utf-8 -*-
"""
Trailroom Routes (backward-compatible)

Endpoints
- GET  /escape/today
- GET  /escape/api/today
- POST /escape/api/submit
- POST /escape/api/finish
- GET  /escape/leaderboard
- GET  /escape/api/leaderboard
- GET  /escape/admin/regen
"""

from __future__ import annotations

import os
import json
import datetime as dt
from typing import Any, Dict
from flask import Blueprint, jsonify, request, render_template, current_app, redirect, url_for, abort

from app.extensions import db
from .models import EscapeAttempt, EscapeRoom, DailyLeaderboardView
from .core import (
    ensure_daily_room,
    verify_puzzle,
    verify_meta_final,
    get_today_key,
)

# ---- Daily Trail chips (non-monetary) ----
CHIPS_START = 100  # starting pool
ROUTE_COST = {"cautious": 0, "brisk": 10}  # "move quickly" costs chips
HINT_COST1 = 10
HINT_COST2 = 20
TIME_BONUS_PER_CHIP_MS = 50  # subtract 50 ms per remaining chip at finish

# ---------------------------------------------------------------------
# Blueprint initializer
# ---------------------------------------------------------------------

def init_routes(bp: Blueprint):
    """
    Attach all route handlers to the provided blueprint.
    """

    # -----------------------------
    # HTML: Play today's room
    # -----------------------------
    @bp.route("/today", methods=["GET"])
    def play_today():
        room = ensure_daily_room()
        return render_template("play.html", date_key=room.date_key, difficulty=room.difficulty)

    # -----------------------------
    # API: Fetch today's room JSON (solutions stripped)
    # -----------------------------
    @bp.route("/api/today", methods=["GET"])
    def api_today():
        date_q = request.args.get("date")
        if date_q:
            existing = db.session.query(EscapeRoom).filter_by(date_key=date_q).first()
            room = existing or ensure_daily_room(date_q)
        else:
            room = ensure_daily_room()

        payload = _strip_solutions(room.json_blob or {})

        # Safety: if something odd produced zero puzzles, rebuild offline once
        if not (payload.get("puzzles") or (payload.get("trail") or {}).get("rooms")):
            current_app.logger.warning("[escape] api_today saw empty content; attempting offline rebuild")
            from .core import generate_room_offline
            secret = os.getenv("ESCAPE_SERVER_SECRET", "dev_secret_change_me")
            try:
                rebuilt = generate_room_offline(room.date_key, secret)
                room.json_blob = rebuilt
                room.difficulty = rebuilt.get("difficulty", room.difficulty)
                db.session.add(room)
                db.session.commit()
                payload = _strip_solutions(rebuilt)
            except Exception as e:
                current_app.logger.error(f"[escape] offline regen failed inside api_today: {e}")

        payload["server_hint_policy"] = {"first_hint_delay_s": 60, "second_hint_delay_s": 120}
        return jsonify(payload)

    # -----------------------------
    # API: Submit an answer to a specific puzzle
    # -----------------------------
    @bp.route("/api/submit", methods=["POST"])
    def api_submit():
        """
        Body: { "date_key": "...", "puzzle_id": "pz_1", "answer": "..." }
        """
        data = _json_body_or_400()
        date_key = data.get("date_key")
        puzzle_id = data.get("puzzle_id")
        answer = data.get("answer")

        if not date_key or not puzzle_id or answer is None:
            return _bad("Missing required fields: date_key, puzzle_id, answer")

        room = _get_or_404(date_key)
        ok = verify_puzzle(room.json_blob, puzzle_id, answer)
        return jsonify({"correct": bool(ok)})

    # -----------------------------
    # API: Finish a run (for leaderboards)
    # -----------------------------
@bp.route("/api/finish", methods=["POST"])
def api_finish():
    data = _json_body_or_400()
    date_key = data.get("date_key")
    started_ms = data.get("started_ms")
    success = bool(data.get("success"))
    meta_client = data.get("meta") or {}

    if not date_key or started_ms is None:
        return _bad("Missing required fields: date_key, started_ms")

    room = _get_or_404(date_key)

    # Duration (server computed)
    now = dt.datetime.utcnow()
    try:
        started_dt = dt.datetime.utcfromtimestamp(int(started_ms) / 1000.0)
    except Exception:
        return _bad("started_ms must be epoch milliseconds")
    dur_ms = int((now - started_dt).total_seconds() * 1000)
    if dur_ms < 3000:
        current_app.logger.info("[escape] suspicious finish: duration < 3s")
        success = False

    # Final-code check (if required)
    if success and room.json_blob.get("final_code") or (room.json_blob.get("final") or {}).get("solution"):
        submitted_final = (data.get("final_code") or "").strip()
        if not submitted_final or not verify_meta_final(room.json_blob, submitted_final):
            success = False

    # ---- Chip math (server-authoritative) ----
    # Client sends which routes they chose and which hints they used per scene.
    # We recompute chips_remaining from the cost table above.
    routes = meta_client.get("routes") or []         # e.g. ["cautious","brisk","cautious"]
    hints_used = meta_client.get("hints") or []      # e.g. [{"h1":true,"h2":false}, ... x3]

    # Coerce shapes
    try:
        routes = [str(x) for x in routes][:3]
    except Exception:
        routes = []
    norm_hints = []
    for h in hints_used[:3]:
        h1 = bool((h or {}).get("h1"))
        h2 = bool((h or {}).get("h2"))
        norm_hints.append({"h1": h1, "h2": h2})
    while len(norm_hints) < 3:
        norm_hints.append({"h1": False, "h2": False})

    # Compute spend
    chips_spent = 0
    for rid in routes[:3]:
        chips_spent += int(ROUTE_COST.get(rid, 0))
    for h in norm_hints[:3]:
        if h.get("h1"): chips_spent += HINT_COST1
        if h.get("h2"): chips_spent += HINT_COST2

    chips_remaining = max(0, min(CHIPS_START - chips_spent, CHIPS_START))
    bonus_ms = chips_remaining * TIME_BONUS_PER_CHIP_MS
    adjusted_ms = max(0, dur_ms - bonus_ms) if success else None

    attempt = EscapeAttempt(
        user_id=None,
        date_key=date_key,
        started_at=started_dt,
        finished_at=now,
        time_ms=adjusted_ms,
        success=success,
        meta={
            **meta_client,
            "chips_start": CHIPS_START,
            "chips_spent": chips_spent,
            "chips_remaining": chips_remaining,
            "bonus_ms": bonus_ms,
            "raw_time_ms": dur_ms,
            "routes": routes,
            "hints": norm_hints,
        },
    )
    db.session.add(attempt)
    db.session.commit()

    return jsonify({
        "ok": True,
        "success": success,
        "time_ms": attempt.time_ms,          # adjusted time (shown on leaderboard)
        "raw_time_ms": dur_ms,
        "bonus_ms": bonus_ms,
        "chips_remaining": chips_remaining,
        "attempt_id": attempt.id,
    })

    # -----------------------------
    # Admin: force regenerate
    # -----------------------------
    @bp.route("/admin/regen", methods=["GET", "POST"])
    def admin_regen():
        from .core import ensure_daily_room, get_today_key

        token = request.args.get("token") or request.headers.get("X-Escape-Admin")
        server_token = os.getenv("ESCAPE_ADMIN_TOKEN", "")
        if not server_token:
            return jsonify({"ok": False, "error": "server missing ESCAPE_ADMIN_TOKEN env"}), 500
        if token != server_token:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

        date_key = request.args.get("date") or get_today_key()
        force = (request.args.get("force", "true").lower() != "false")
        try:
            row = ensure_daily_room(date_key, force_regen=force)
            puzzles = len(((row.json_blob or {}).get("puzzles") or []))
            return jsonify({"ok": True, "date": date_key, "puzzles": puzzles})
        except Exception as e:
            current_app.logger.exception("[escape] admin_regen failed")
            return jsonify({"ok": False, "error": str(e)}), 500

    # -----------------------------
    # HTML: Leaderboard
    # -----------------------------
    @bp.route("/leaderboard", methods=["GET"])
    def leaderboard_html():
        date_q = request.args.get("date") or get_today_key()
        rows = DailyLeaderboardView.top_for_day(date_q, limit=50)
        top = []
        for r in rows:
            a = r["attempt"]
            top.append({
                "rank": r["rank"],
                "time_ms": a.time_ms,
                "seconds": (a.time_ms / 1000.0) if a.time_ms else None,
                "created_at": a.created_at,
            })
        return render_template("escape/leaderboard.html", date_key=date_q, rows=top)

    # -----------------------------
    # API: Leaderboard JSON
    # -----------------------------
    @bp.route("/api/leaderboard", methods=["GET"])
    def leaderboard_api():
        date_q = request.args.get("date") or get_today_key()
        try:
            limit = int(request.args.get("limit", "50"))
        except Exception:
            limit = 50
        limit = max(1, min(200, limit))

        rows = DailyLeaderboardView.top_for_day(date_q, limit=limit)
        out = []
        for r in rows:
            a = r["attempt"]
            out.append({
                "rank": r["rank"],
                "time_ms": a.time_ms,
                "seconds": (a.time_ms / 1000.0) if a.time_ms else None,
                "created_at": a.created_at.isoformat() + "Z",
            })
        return jsonify({"date_key": date_q, "top": out})

    # Root -> /today
    @bp.route("/", methods=["GET"])
    def root():
        return redirect(url_for("escape.play_today"), code=302)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# app/escape/routes.py

def _strip_solutions(room_json):
    if not isinstance(room_json, dict):
        return {}
    out = json.loads(json.dumps(room_json))  # deep copy

    # 1) flat list (legacy)
    cleaned = []
    for p in out.get("puzzles", []):
        if isinstance(p, dict):
            p.pop("solution", None)
            cleaned.append(p)
    out["puzzles"] = cleaned

    # 2) trail shape
    trail = out.get("trail") or {}
    for rm in (trail.get("rooms") or []):
        for rt in (rm.get("routes") or []):
            p = rt.get("puzzle")
            if isinstance(p, dict):
                p.pop("solution", None)
    out["trail"] = trail

    return out

def _json_body_or_400() -> Dict[str, Any]:
    if not request.data:
        abort(_abort_json(400, "Request body required"))
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        abort(_abort_json(400, "Invalid JSON"))
    if data is None:
        abort(_abort_json(400, "Request body required"))
    return data

def _abort_json(status: int, message: str):
    resp = jsonify({"ok": False, "error": message})
    resp.status_code = status
    return resp

def _bad(message: str):
    return _abort_json(400, message)

def _get_or_404(date_key: str) -> EscapeRoom:
    try:
        existing = db.session.query(EscapeRoom).filter_by(date_key=date_key).first()
        room = existing or ensure_daily_room(date_key)
        return room
    except Exception as e:
        current_app.logger.error(f"[escape] unable to load room for {date_key}: {e}")
        abort(_abort_json(404, "Room not found"))
