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
    apply_fragment_rule,   # ← NEW: compute fragment server-side
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
        frag = _fragment_for_submission(room.json_blob, puzzle_id, answer) if ok else None
        return jsonify({"correct": bool(ok), "fragment": frag})

    # -----------------------------
    # API: Finish a run (for leaderboards)
    # -----------------------------
    @bp.route("/api/finish", methods=["POST"])
    def api_finish():
        """
        Record a player's run. Duration is computed server-side.
        We apply (a) leftover-chips bonus and (b) route risk/reward bonus/penalty.
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

        # Verify final meta-code if provided/required
        if success and (room.json_blob.get("final_code") or (room.json_blob.get("final") or {}).get("solution")):
            submitted_final = (data.get("final_code") or "").strip()
            if not submitted_final or not verify_meta_final(room.json_blob, submitted_final):
                success = False

        eff_ms = dur_ms
        chip_bonus_ms = 0
        route_bonus_ms = 0
        route_penalty_ms = 0

        if success:
            # Apply bonuses in sequence on the running effective time
            eff_ms, chip_bonus_ms = _apply_chip_bonus(eff_ms, meta)
            eff_ms, route_bonus_ms, route_penalty_ms = _apply_approach_bonus(eff_ms, meta)

        # Save attempt
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
                "chip_bonus_ms": int(chip_bonus_ms),
                "route_bonus_ms": int(route_bonus_ms),
                "route_penalty_ms": int(route_penalty_ms),
                "total_bonus_ms": int(chip_bonus_ms + route_bonus_ms - route_penalty_ms),
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

def _fragment_for_submission(room_json: Dict[str, Any], puzzle_id: str, submitted_answer: str) -> Optional[str]:
    """
    Find the room containing puzzle_id, get its fragment_rule,
    and compute the fragment from the *submitted* answer.
    CONST: rules remain constant; for others we apply FIRST2/LAST2/etc.
    """
    trail = room_json.get("trail") or {}
    for rm in (trail.get("rooms") or []):
        rule = (rm.get("fragment_rule") or "FIRST2")
        for rt in (rm.get("routes") or []):
            pz = (rt.get("puzzle") or {})
            if pz.get("id") == puzzle_id:
                try:
                    return apply_fragment_rule(submitted_answer, rule)
                except Exception:
                    return None
    return None

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

def _apply_approach_bonus(dur_ms: int, meta: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Risk/reward per scene.
    Supports both old route ids ("cautious","brisk","risky") and new labels
    ("observe","listen","traverse"). Bonus only if (no hints AND first try).
    Penalty if >=3 tries.
    """
    routes = meta.get("routes") or []
    hints  = meta.get("hints_used_scene") or []
    tries  = meta.get("submissions_scene") or []

    MAP = {"cautious": "observe", "brisk": "listen", "risky": "traverse"}
    TUNE = {
        "observe":  {"bonus": 0,    "penalty": 0},
        "listen":   {"bonus": 1500, "penalty": 2000},
        "traverse": {"bonus": 3000, "penalty": 5000},
    }

    bonus = penalty = 0
    for i, r in enumerate(routes[:3]):
        key = MAP.get((r or "").lower(), (r or "").lower())
        h = int(hints[i]) if i < len(hints) else 0
        t = int(tries[i]) if i < len(tries) else 0
        tune = TUNE.get(key, {"bonus": 0, "penalty": 0})
        if h == 0 and t == 1:
            bonus += tune["bonus"]
        elif t >= 3:
            penalty += tune["penalty"]

    eff = max(0, dur_ms - bonus + penalty)
    return eff, bonus, penalty

def _apply_chip_bonus(dur_ms: int, meta: Dict[str, Any]) -> Tuple[int, int]:
    """
    Convert leftover chips into a time bonus.
    Expects client to send `chips_left` (or `chips`) in meta.
    """
    chips_left = int((meta or {}).get("chips_left", (meta or {}).get("chips", 0)) or 0)
    chips_left = max(0, min(60, chips_left))   # sanity clamp
    MS_PER_CHIP = 400                          # tune: 0.4s per chip
    bonus = chips_left * MS_PER_CHIP
    eff = max(0, dur_ms - bonus)
    return eff, bonus
