# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Routes (Blueprint endpoints)
"""

from __future__ import annotations

import os
import json
import re
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import os, hashlib, secrets
from random import Random
from flask import Blueprint, jsonify, request, render_template, current_app, redirect, url_for, abort, session

from app.extensions import db
from app.subscriptions import current_user_has_plus
from .models import EscapeAttempt, EscapeRoom, DailyLeaderboardView
from .core import (
    ensure_daily_room,
    verify_puzzle,
    verify_meta_final,
    get_today_key,
    attach_daily_minis,   # <-- we will call this correctly
)
from app.daily_status import ESCAPE_COMPLETE_KEY

def _rng_for_date(date_key: str) -> Random:
    """Deterministic RNG per date using server secret (+ optional salt)."""
    secret = os.getenv("ESCAPE_SERVER_SECRET", "dev_secret_change_me")
    salt = os.getenv("ESCAPE_REGEN_SALT", "")
    h = hashlib.sha256(f"{date_key}|{secret}|{salt}".encode()).digest()
    seed = int.from_bytes(h[:4], "little")
    return Random(seed)

# Optional: fragment rule import (fallback if core lacks it)
try:
    from .core import apply_fragment_rule  # computes per-scene fragment from submitted answer
except Exception:  # pragma: no cover
    def apply_fragment_rule(answer: str, rule: str) -> Optional[str]:
        s = (answer or "").strip()
        r = (rule or "").upper()
        if not s:
            return None
        return {"FIRST2": s[:2], "LAST2": s[-2:], "FIRST3": s[:3], "LAST3": s[-3:]}.get(r, s[:2])


# ---------------------------------------------------------------------
# Helpers (pure functions)
# ---------------------------------------------------------------------

def _row_to_blob(row):
    return row.json_blob or {}

def _blob_to_minis_payload(blob, date_key):
    """
    Normalize the room blob to {date,theme,minigames:[]} for API.
    Tolerates dicts, Puzzle objects, or missing minis (attaches them).
    Uses the SAME per-day RNG as /api/today so results never drift.
    """
    from .core import attach_daily_minis

    def _to_dict(x):
        if isinstance(x, dict):
            return x
        if hasattr(x, "to_json"):
            try:
                return x.to_json()
            except Exception:
                return {}
        return dict(getattr(x, "__dict__", {}) or {})

    mg = blob.get("minigames")
    if not mg:
        # backfill deterministically with our unified RNG path
        rng = _rng_for_date(date_key)
        attach_daily_minis(blob, rng, blob.get("theme") or blob.get("title") or "Daily Escape")
        mg = blob.get("minigames")

    minis = []
    for m in (mg or []):
        md = _to_dict(m)
        if not md:
            continue
        if not md.get("puzzle_id"):
            md["puzzle_id"] = md.get("id")
        md["mechanic"] = (md.get("mechanic") or "").lower()
        minis.append(md)

    return {
        "date": blob.get("date") or blob.get("date_key") or date_key,
        "theme": blob.get("theme") or blob.get("title") or "Daily Escape",
        "minigames": minis,
    }

def _has_any_puzzle(room_json: Dict[str, Any]) -> bool:
    if (room_json.get("puzzles") or []):
        return True
    trail = room_json.get("trail") or {}
    for rm in (trail.get("rooms") or []):
        # room-level
        if isinstance(rm.get("puzzle"), dict) and rm["puzzle"]:
            return True
        for rt in (rm.get("routes") or []):
            if isinstance(rt, dict) and (rt.get("puzzle") or {}):
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
        # room-level
        if isinstance(rm.get("puzzle"), dict):
            rm["puzzle"].pop("solution", None)
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
    """Compute per-scene fragment from submitted answer using each room's rule."""
    trail = room_json.get("trail") or {}
    for rm in (trail.get("rooms") or []):
        rule = (rm.get("fragment_rule") or "FIRST2")
        # room-level mini
        p_room = rm.get("puzzle") or {}
        if p_room.get("id") == puzzle_id:
            try:
                return apply_fragment_rule(submitted_answer, rule)
            except Exception:
                return None
        # route-level
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
    Apply time adjustments based on chosen route per scene.
    meta: route_ids or routes (synonyms allowed), hints_used_scene, submissions_scene
    """
    raw_routes = meta.get("route_ids") or meta.get("routes") or []

    def _norm(val: Any) -> str:
        v = str(val or "").lower().strip()
        if ("caut" in v) or ("care" in v) or ("observe" in v) or ("inspect" in v) or ("safe" in v):
            return "cautious"
        if ("brisk" in v) or ("quick" in v) or ("fast" in v) or ("listen" in v) or ("weave" in v):
            return "brisk"
        if ("risk" in v) or ("bold" in v) or ("traverse" in v) or ("tamper" in v):
            return "risky"
        return v

    routes = [_norm(r) for r in list(raw_routes)[:3]]
    hints  = list(meta.get("hints_used_scene") or [])
    tries  = list(meta.get("submissions_scene") or [])

    TUNE = {
        "cautious": {"bonus": 0,    "penalty": 0},
        "brisk":    {"bonus": 1500, "penalty": 2000},
        "risky":    {"bonus": 3000, "penalty": 5000},
    }

    bonus = 0
    penalty = 0
    for i, r in enumerate(routes):
        tune = TUNE.get(r, {"bonus": 0, "penalty": 0})
        h = int(hints[i]) if i < len(hints) else 0
        t = int(tries[i]) if i < len(tries) else 0
        if h == 0 and t == 1:
            bonus += tune["bonus"]
        elif t >= 3:
            penalty += tune["penalty"]

    eff = max(0, int(dur_ms) - bonus + penalty)
    return eff, bonus, penalty

def _apply_chip_bonus(dur_ms: int, meta: Dict[str, Any]) -> Tuple[int, int]:
    """Convert leftover chips into a time bonus."""
    chips_left = int((meta or {}).get("chips_left", (meta or {}).get("chips", 0)) or 0)
    chips_left = max(0, min(60, chips_left))
    MS_PER_CHIP = 400
    bonus = chips_left * MS_PER_CHIP
    eff = max(0, dur_ms - bonus)
    return eff, bonus


# ---------------------------------------------------------------------
# Blueprint initializer (idempotent)
# ---------------------------------------------------------------------

def init_routes(bp: Blueprint):
    """Attach all route handlers to the provided blueprint."""
    if getattr(bp, "_escape_inited", False):
        return bp
    bp._escape_inited = True

    # HTML: Play (support /today and /play for back-compat)
    @bp.route("/today", methods=["GET"])
    def play_today():
        room = ensure_daily_room()
        return render_template(
            "escape/play.html",
            date_key=room.date_key,
            difficulty=room.difficulty,
            has_plus=current_user_has_plus(),
        )

    @bp.route("/play", methods=["GET"])
    def play_alias():
        return redirect(url_for("escape.play_today"), code=302)

    @bp.route("/api/admin/regen", methods=["GET", "POST"])
    def admin_regen_api():
        token = request.args.get("token") or (request.get_json(silent=True) or {}).get("token")
        if token != os.getenv("ESCAPE_ADMIN_TOKEN"):
            return jsonify({"ok": False, "error": "forbidden"}), 403

        date_key = request.args.get("date") or get_today_key()
        force = (str(request.args.get("force") or "").lower() in ("1","true","yes","y"))

        # Prefer explicit salt/rotate; otherwise auto-generate a one-time salt when forcing
        salt = request.args.get("salt") or request.args.get("rotate")
        auto_salt = False
        if force and not salt:
            import secrets
            salt = f"auto-{secrets.token_hex(4)}"
            auto_salt = True

        old = os.environ.get("ESCAPE_REGEN_SALT")
        try:
            if salt:
                os.environ["ESCAPE_REGEN_SALT"] = salt
            row = ensure_daily_room(date_key, force_regen=force)

            # ⤵ Persist minis NOW (while rotate/salt is active) so future reads match.
            try:
                from .core import attach_daily_minis
                rng = _rng_for_date(date_key)
                blob = _row_to_blob(row)
                theme = blob.get("title") or blob.get("theme") or "Daily Escape"
                # Regenerate minis when forced, or create if missing
                if force or not (blob.get("minigames") or []):
                    attach_daily_minis(blob, rng, theme)
                    row.json_blob = blob
                    db.session.add(row)
                    db.session.commit()
            except Exception as e:
                current_app.logger.warning(f"[escape] admin_regen minis persist failed: {e}")

        finally:
            if salt:
                if old is None:
                    os.environ.pop("ESCAPE_REGEN_SALT", None)
                else:
                    os.environ["ESCAPE_REGEN_SALT"] = old

        # Include salt details for observability
        return jsonify({"ok": True, "date": row.date_key, "rotate": salt, "auto_rotate": auto_salt})

    @bp.route("/admin/ping", methods=["GET"])
    def admin_ping():
        return jsonify({"ok": True, "blueprint": bp.name})

    # API: today (legacy blob by default; minis view with ?format=minis)
    @bp.route("/api/today", methods=["GET"])
    def api_today():
        fmt = (request.args.get("format") or "").lower()
        date_q = request.args.get("date")
        if date_q:
            existing = db.session.query(EscapeRoom).filter_by(date_key=date_q).first()
            room = existing or ensure_daily_room(date_q)
        else:
            room = ensure_daily_room()
        blob = _row_to_blob(room)

        # Ensure new-style minis exist; attach if the blob predates minis.
        if not (blob.get("minigames") or []):
            try:
                rng = _rng_for_date(room.date_key)
                theme = blob.get("title") or blob.get("theme") or "Daily Escape"
                attach_daily_minis(blob, rng, theme)   # mutate in place
                # ⤵ persist once so future reads don't re-generate different minis
                room.json_blob = blob
                db.session.add(room)
                db.session.commit()
            except Exception as e:
                current_app.logger.warning(f"[escape] attach_daily_minis failed: {e}")

        if fmt == "minis":
            payload = _blob_to_minis_payload(blob, room.date_key)
            if not payload.get("minigames"):
                current_app.logger.warning("[escape] minis view built 0 games for %s", room.date_key)
            return jsonify(payload)

        payload = _strip_solutions(blob)

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

    @bp.route("/api/submit", methods=["POST"])
    def api_submit():
        from .core import attach_daily_minis  # for legacy rows without minis
        data = _json_body_or_400()
        # accept both "date" and "date_key"
        date_key = (data.get("date") or data.get("date_key") or get_today_key()).strip()
        puzzle_id = (data.get("puzzle_id") or "").strip()
        answer_raw = (data.get("answer") or "").strip()
        if not puzzle_id:
            return _bad("Missing puzzle_id")

        # load the day blob the verifier expects
        room = _get_or_404(date_key)
        blob = _row_to_blob(room)

        # Backfill minis deterministically if this row predates minis
        if not (blob.get("minigames") or []):
            try:
                rng = _rng_for_date(date_key)
                theme = blob.get("title") or blob.get("theme") or "Daily Escape"
                attach_daily_minis(blob, rng, theme)
            except Exception as e:
                current_app.logger.warning(f"[escape] submit backfill minis failed: {e}")

        # locate the mini and mechanic
        def _find_mini(pid):
            for m in (blob.get("minigames") or []):
                if (m.get("puzzle_id") or m.get("id")) == pid:
                    return m
            return None

        mini = _find_mini(puzzle_id) or {}
        ui  = (mini.get("ui_spec") or mini.get("ui") or {})
        mech = (mini.get("mechanic") or "").lower()

        # Canonicalize Vault Frenzy **to indices** (server stores indices)
        def _vf_to_indices(ans: str) -> str:
            tokens = [t.strip() for t in str(ans).replace(";", ",").split(",") if t.strip()]
            if not tokens: return ""
            if all(t.isdigit() for t in tokens):
                return ",".join(str(int(t)) for t in tokens)
            cols = ((ui.get("grid") or {}) or {}).get("cols") or ui.get("grid_cols") or ui.get("gridCols") or 4
            out = []
            for t in tokens:
                if "-" in t:
                    a, b = t.split("-", 1)
                    try:
                        r, c = int(a), int(b)
                        out.append(str(r * int(cols) + c))
                    except Exception:
                        pass
            return ",".join(out)

        answer = _vf_to_indices(answer_raw) if mech == "vault_frenzy" else answer_raw

        try:
            correct = bool(verify_puzzle(blob, puzzle_id, answer))
            return jsonify({"ok": True, "correct": correct})
        except Exception as e:
            current_app.logger.exception("submit verify failed: %s", e)
            return jsonify({"ok": False, "correct": False, "error": "verify_failed"}), 200

    # API: finish a run (leaderboards)
    @bp.route("/api/finish", methods=["POST"])
    def api_finish():
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

        # Optional final-code verification (opt-in only).
        # We only enforce a final code if the caller explicitly asks us to.
        blob_final = room.json_blob.get("final_code") or (room.json_blob.get("final") or {}).get("solution")
        enforce_final = bool((data.get("enforce_final") is True) or ((data.get("meta") or {}).get("enforce_final") is True))
        if success and blob_final and enforce_final:
            submitted_final = (data.get("final_code") or "").strip()
            if not submitted_final or not verify_meta_final(room.json_blob, submitted_final):
                success = False

        eff_ms = dur_ms
        chip_bonus_ms = 0
        route_bonus_ms = 0
        route_penalty_ms = 0

        if success:
            eff_ms, chip_bonus_ms = _apply_chip_bonus(eff_ms, meta)
            eff_ms, route_bonus_ms, route_penalty_ms = _apply_approach_bonus(eff_ms, meta)

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
        if success:
            try:
                session[ESCAPE_COMPLETE_KEY] = date_key
            except Exception:
                pass

        return jsonify({
            "ok": True,
            "success": success,
            "time_ms": attempt.time_ms,
            "attempt_id": attempt.id,
        })

    # HTML leaderboard
    @bp.route("/leaderboard", methods=["GET"])
    def leaderboard_html():
        # accept ?date or ?date_key; default to today's key
        date_q = request.args.get("date") or request.args.get("date_key") or get_today_key()
        rows = DailyLeaderboardView.top_for_day(date_q, limit=50)

        # Normalize rows to what the template expects
        top = []
        for r in rows:
            a = r["attempt"]
            top.append({
                "rank": r["rank"],
                "nickname": (a.meta or {}).get("nickname") or "Guest",
                "total_time_ms": int(a.time_ms or 0),
                "finished_iso": (a.created_at.isoformat() + "Z") if getattr(a, "created_at", None) else None,
            })

        # NOTE: template reads {{ date }} and iterates {{ rows }}
        return render_template("escape/leaderboard.html", date=date_q, rows=top)

    # JSON leaderboard (also normalized)
    @bp.route("/api/leaderboard", methods=["GET"])
    def leaderboard_api():
        date_q = request.args.get("date") or request.args.get("date_key") or get_today_key()
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
                "nickname": (a.meta or {}).get("nickname") or "Guest",
                "total_time_ms": int(a.time_ms or 0),
                "finished_iso": (a.created_at.isoformat() + "Z") if getattr(a, "created_at", None) else None,
            })
        return jsonify({"date": date_q, "rows": out})

    # Root
    @bp.route("/", methods=["GET"])
    def root():
        return redirect(url_for("escape.play_today"), code=302)

    return bp
