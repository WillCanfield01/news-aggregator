# app/escape/routes.py
# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Routes (Blueprint endpoints)

Endpoints
- GET  /escape/today                 -> HTML page to play
- GET  /escape/api/today             -> JSON room (solutions stripped)
- POST /escape/api/submit            -> {date_key, puzzle_id, answer} -> {correct: bool}
- POST /escape/api/finish            -> {date_key, started_ms, success, meta?, final_code?} -> leaderboard save
- GET  /escape/leaderboard           -> HTML page with today's top times
- GET  /escape/api/leaderboard       -> JSON leaderboard {date_key?, limit?}

Notes
- Keep routes thin; generation/validation lives in core.py, DB in models.py.
- Solutions are *never* sent to the client. Only server verifies answers.
"""

from __future__ import annotations

import time
import math
import datetime as dt
from typing import Any, Dict, List, Optional

from flask import (
    request,
    jsonify,
    render_template,
    current_app,
)
from app.extensions import db
from .models import EscapeAttempt
from .models import EscapeRoom
from .models import DailyLeaderboardView
from .core import (
    ensure_daily_room,
    verify_puzzle,
    verify_meta_final,
    get_today_key,
)

# ---------------------------------------------------------------------
# Blueprint initializer
# ---------------------------------------------------------------------

def init_routes(bp):
    """
    Attach all route handlers to the provided blueprint.
    """

    # -----------------------------
    # HTML: Play today's room
    # -----------------------------
    @bp.route("/today", methods=["GET"])
    def play_today():
        """
        Render the single-page UI. The template fetches the room via /api/today.
        """
        # Ensure today's room exists (generation is idempotent).
        room = ensure_daily_room()
        # Pass only minimal info to template; the JS will fetch /api/today
        return render_template("play.html", date_key=room.date_key, difficulty=room.difficulty)


    # -----------------------------
    # API: Fetch today's room JSON (solutions stripped)
    # -----------------------------
    @bp.route("/api/today", methods=["GET"])
    def api_today():
        """
        Return the JSON for today's room with solutions stripped off.
        Optional ?date=YYYY-MM-DD to fetch historical day (if exists and you allow archives later).
        """
        date_q = request.args.get("date")
        if date_q:
            # Allow fetching an existing day if in DB; otherwise default to today's
            existing = db.session.query(EscapeRoom).filter_by(date_key=date_q).first()
            room = existing or ensure_daily_room(date_q)
        else:
            room = ensure_daily_room()

        payload = _strip_solutions(room.json_blob)
        # Attach server-enforced timing hints (client may respect, but server-side anti-cheat also checks)
        payload["server_hint_policy"] = {"first_hint_delay_s": 60, "second_hint_delay_s": 120}
        return jsonify(payload)

    # -----------------------------
    # API: Submit an answer to a specific puzzle
    # -----------------------------
    @bp.route("/api/submit", methods=["POST"])
    def api_submit():
        """
        Validate a single puzzle answer.
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
        """
        Record a player's run. Minimal anti-cheat: duration sanity + optional final_code check (if enabled).
        Body:
        {
          "date_key": "...",
          "started_ms": 1693584000000,    # epoch ms
          "success": true,
          "final_code": "AB12"            # required if room has meta final_code
          "meta": { "ua": "...", "coarse_region": "US", ... }  # optional client hints
        }
        """
        data = _json_body_or_400()
        date_key = data.get("date_key")
        started_ms = data.get("started_ms")
        success = bool(data.get("success"))
        meta = data.get("meta") or {}

        if not date_key or started_ms is None:
            return _bad("Missing required fields: date_key, started_ms")

        # Look up the room; generate if needed (idempotent)
        room = _get_or_404(date_key)

        # Server computes duration; client-sent finished time is ignored
        now = dt.datetime.utcnow()
        try:
            started_dt = dt.datetime.utcfromtimestamp(int(started_ms) / 1000.0)
        except Exception:
            return _bad("started_ms must be epoch milliseconds")

        # Anti-cheat: minimum time sanity (e.g., at least 3 seconds)
        dur_ms = int((now - started_dt).total_seconds() * 1000)
        if dur_ms < 3000:
            current_app.logger.info("[escape] suspicious finish: duration < 3s")
            # Treat as non-successful
            success = False

        # If meta final_code is required, verify (when 'success' claims true)
        if success and room.json_blob.get("final_code"):
            submitted_final = (data.get("final_code") or "").strip()
            if not submitted_final or not verify_meta_final(room.json_blob, submitted_final):
                # Final lock not satisfied; do not count as success
                success = False

        attempt = EscapeAttempt(
            user_id=None,
            date_key=date_key,
            started_at=started_dt,
            finished_at=now,
            time_ms=dur_ms if success else None,
            success=success,
            meta=meta,
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
        """
        Super simple leaderboard page for today's top times.
        (Your template can iterate the rows; this is just a helper route.)
        """
        date_q = request.args.get("date") or get_today_key()
        rows = DailyLeaderboardView.top_for_day(date_q, limit=50)
        # Convert attempts to dicts for rendering convenience
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
        """
        Query string:
          - date (optional): YYYY-MM-DD; defaults to today's date
          - limit (optional): default 50
        """
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _strip_solutions(room_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deep copy with solution data removed so we never leak answers to the client.
    """
    # Shallow copy of top-level
    out = dict(room_json)
    # Deep-ish copy for puzzles stripping solution
    puzzles = []
    for p in room_json.get("puzzles", []):
        pp = dict(p)
        pp.pop("solution", None)
        puzzles.append(pp)
    out["puzzles"] = puzzles
    # Do not remove final_code existence; client may need to know a meta gate exists,
    # but we do not expose its value (only existence). The server verifies it on finish.
    return out


def _json_body_or_400() -> Dict[str, Any]:
    """
    Parse JSON or return a 400 response.
    """
    if not request.data:
        return _abort_json(400, "Request body required")
    try:
        return request.get_json(force=True, silent=False) or {}
    except Exception:
        return _abort_json(400, "Invalid JSON")


def _abort_json(status: int, message: str):
    """
    Return an error JSON with the provided status code.
    """
    resp = jsonify({"ok": False, "error": message})
    resp.status_code = status
    return resp


def _bad(message: str):
    """
    Shortcut for 400 errors.
    """
    return _abort_json(400, message)


def _get_or_404(date_key: str) -> EscapeRoom:
    """
    Fetch an EscapeRoom by date_key or generate it. If generation fails, return 404.
    """
    try:
        # If it exists in DB, use it; otherwise ensure generation
        existing = db.session.query(EscapeRoom).filter_by(date_key=date_key).first()
        return existing or ensure_daily_room(date_key)
    except Exception as e:
        current_app.logger.error(f"[escape] unable to load room for {date_key}: {e}")
        return _abort_json(404, "Room not found")
