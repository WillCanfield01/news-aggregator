# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Routes (Blueprint endpoints)

Endpoints
- GET  /escape/today                 -> HTML page to play
- GET  /escape/api/today             -> JSON room (solutions stripped)
- POST /escape/api/submit            -> {date_key, puzzle_id, answer} -> {correct: bool}
- POST /escape/api/finish            -> {date_key, started_ms, success, final_code?, meta?} -> leaderboard save
- GET  /escape/leaderboard           -> HTML page with today's top times
- GET  /escape/api/leaderboard       -> JSON leaderboard {date_key?, limit?}

Notes
- Keep routes thin; generation/validation lives in core.py, DB in models.py.
- Solutions are *never* sent to the client. Only server verifies answers.
"""

from __future__ import annotations

import os
import json
import time
import math
import datetime as dt
from typing import Any, Dict, List, Optional
from typing import Tuple
from flask import Blueprint, jsonify, request, render_template, current_app, redirect, url_for, abort

from app.extensions import db
from .models import EscapeAttempt, EscapeRoom, DailyLeaderboardView
from .core import (
    ensure_daily_room,
    verify_puzzle,
    verify_meta_final,
    get_today_key,
)

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
        return render_template("escape/play.html", date_key=room.date_key, difficulty=room.difficulty)

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

    # --- Admin health & alias ---
    @bp.route("/admin/ping", methods=["GET"])
    def admin_ping():
        return jsonify({"ok": True, "blueprint": bp.name})

    # Keep /admin/regen, add an API alias so we have two ways to hit it
    @bp.route("/api/admin/regen", methods=["GET", "POST"])
    def admin_regen_api():
        return admin_regen()   # just call the handler below

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

        # If something went wrong and we ended up with 0 puzzles/routes, hard fallback to offline
        if not _has_any_puzzle(payload):
            current_app.logger.warning("[escape] api_today saw no puzzles/routes; attempting offline regen")
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
        """
        Record a player's run. We compute the duration server-side.
        If meta includes chip info, we apply a time bonus (leftover chips -> shorter effective time).
        """
        data = _json_body_or_400()
        date_key = data.get("date_key")
        started_ms = data.get("started_ms")
        success = bool(data.get("success"))
        meta = data.get("meta") or {}

        if not date_key or started_ms is None:
            return _bad("Missing required fields: date_key, started_ms")

        room = _get_or_404(date_key)

        now = dt.datetime.utcnow()
        try:
            started_dt = dt.datetime.utcfromtimestamp(int(started_ms) / 1000.0)
        except Exception:
            return _bad("started_ms must be epoch milliseconds")

        dur_ms = int((now - started_dt).total_seconds() * 1000)
        if dur_ms < 3000:
            current_app.logger.info("[escape] suspicious finish: duration < 3s")
            success = False

        # If a final meta-code exists on the room, verify it when success is True
        if success and (room.json_blob.get("final_code") or (room.json_blob.get("final") or {}).get("solution")):
            submitted_final = (data.get("final_code") or "").strip()
            if not submitted_final or not verify_meta_final(room.json_blob, submitted_final):
                success = False

        # Apply server-side chip bonus (only with sane meta)
        # Apply server-side chip + route risk/reward (only with sane meta)
        eff_ms = dur_ms
        if success:
            # This mutates `meta` to include a breakdown:
            # chip_bonus_ms, route_bonus_ms, route_penalty_ms, effective_time_ms
            eff_ms, _ = _apply_chip_bonus(dur_ms, meta)

        # Save attempt (persist the breakdown populated by _apply_chip_bonus)
        attempt = EscapeAttempt(
            user_id=None,
            date_key=date_key,
            started_at=started_dt,
            finished_at=now,
            time_ms=eff_ms if success else None,
            success=success,
            meta={
                **(meta or {}),
                "raw_time_ms": dur_ms,
                # Normalize numbers in case meta is missing anything
                "chip_bonus_ms": int((meta or {}).get("chip_bonus_ms", 0)),
                "route_bonus_ms": int((meta or {}).get("route_bonus_ms", 0)),
                "route_penalty_ms": int((meta or {}).get("route_penalty_ms", 0)),
                "total_bonus_ms": (
                    int((meta or {}).get("chip_bonus_ms", 0))
                    + int((meta or {}).get("route_bonus_ms", 0))
                    - int((meta or {}).get("route_penalty_ms", 0))
                ),
                "effective_time_ms": eff_ms if success else None,
            },
        )

        db.session.add(attempt)
        db.session.commit()

        return jsonify({
            "ok": True,
            "success": success,
            "time_ms": attempt.time_ms,
            "attempt_id": attempt.id,
        })

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

    @bp.route("/", methods=["GET"])
    def root():
        return redirect(url_for("escape.play_today"), code=302)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _has_any_puzzle(room_json: Dict[str, Any]) -> bool:
    if (room_json.get("puzzles") or []):
        return True
    trail = room_json.get("trail") or {}
    for rm in (trail.get("rooms") or []):
        for rt in (rm.get("routes") or []):
            p = (rt.get("puzzle") or {})
            if p:
                return True
    return False

def _strip_solutions(room_json: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy and strip ALL solutions from puzzles and final."""
    if not isinstance(room_json, dict):
        return {}

    out = json.loads(json.dumps(room_json))  # deep copy

    # Legacy flat list
    cleaned = []
    for p in (out.get("puzzles") or []):
        if isinstance(p, dict):
            p.pop("solution", None)
            cleaned.append(p)
    if "puzzles" in out:
        out["puzzles"] = cleaned

    # New trail shape
    trail = out.get("trail") or {}
    for rm in (trail.get("rooms") or []):
        for rt in (rm.get("routes") or []):
            if isinstance(rt, dict):
                pz = rt.get("puzzle")
                if isinstance(pz, dict):
                    pz.pop("solution", None)

    # Final solution (never send)
    fin = out.get("final")
    if isinstance(fin, dict):
        if isinstance(fin.get("solution"), dict):
            fin["solution"] = {}

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
    abort(_abort_json(400, message))

def _get_or_404(date_key: str) -> EscapeRoom:
    try:
        existing = db.session.query(EscapeRoom).filter_by(date_key=date_key).first()
        return existing or ensure_daily_room(date_key)
    except Exception as e:
        current_app.logger.error(f"[escape] unable to load room for {date_key}: {e}")
        return abort(_abort_json(404, "Room not found"))

def _apply_chip_bonus(dur_ms: int, meta: Dict[str, Any]) -> (int, int):
    """
    Risk/Reward with routes + chips -> time conversion.

    Server-enforced:
      - Per-chip conversion depends on difficulty (easy/med/hard).
      - Route bonuses/penalties:
          brisk:  first-try + no hints => -2000ms; 3+ tries => +1500ms
          risky:  first-try + no hints => -3500ms; 3+ tries => +3000ms
      - Cautious has no route-based adjustment.

    Expected meta (all optional; we sanitize):
      chips_start, chips_spent, chips_remaining
      routes: ["cautious"|"brisk"|"risky", ...] length 3
      hints_used_scene: [int,int,int]
      submissions_scene: [int,int,int]
      difficulty: "easy"|"medium"|"hard" (from client is advisory; server can override if needed)
    """
    # --- Chips sanity ---
    try:
        start  = int(meta.get("chips_start", 0))
        spent  = int(meta.get("chips_spent", 0))
        remain = int(meta.get("chips_remaining", 0))
    except Exception:
        start = spent = remain = 0

    if (start < 0 or spent < 0 or remain < 0 or
        remain > start or spent > start or (remain + spent) > start):
        remain = 0  # bad accounting -> no chip bonus

    # Difficulty-scaled per-chip
    diff = (meta.get("difficulty") or "").lower()
    per_chip = 40
    if diff == "easy": per_chip = 30
    elif diff == "hard": per_chip = 50

    chip_bonus_ms = remain * per_chip

    # --- Route risk/reward ---
    routes = meta.get("routes") or []
    hints  = meta.get("hints_used_scene") or []
    tries  = meta.get("submissions_scene") or []

    route_bonus_ms = 0
    route_penalty_ms = 0

    for i, r in enumerate(routes[:3]):
        rname = str(r or "").lower()
        h = int(hints[i]) if i < len(hints) and str(hints[i]).isdigit() else 0
        t = int(tries[i]) if i < len(tries) and str(tries[i]).isdigit() else 0

        if rname == "brisk":
            if h == 0 and t == 1:
                route_bonus_ms += 2000
            elif t >= 3:
                route_penalty_ms += 1500
        elif rname == "risky":
            if h == 0 and t == 1:
                route_bonus_ms += 3500
            elif t >= 3:
                route_penalty_ms += 3000
        # cautious: no adjustments

    effective_ms = max(0, dur_ms - chip_bonus_ms - route_bonus_ms + route_penalty_ms)
    total_bonus_ms = chip_bonus_ms + route_bonus_ms - route_penalty_ms

    # Keep breakdown (useful for leaderboard audit)
    meta["chip_bonus_ms"]     = chip_bonus_ms
    meta["route_bonus_ms"]    = route_bonus_ms
    meta["route_penalty_ms"]  = route_penalty_ms
    meta["effective_time_ms"] = effective_ms

    return effective_ms, total_bonus_ms
