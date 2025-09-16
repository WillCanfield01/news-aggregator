
# app/escape/core.py (FULL — compatible + new minigames)
# -*- coding: utf-8 -*-
"""
Daily Escape — Core module
- Deterministic daily generation of 3 ORIGINAL minigames
- Server-side verification (no solutions sent to client)
- HMAC-signed mini completion tokens
- Back-compat shims for legacy imports (schedule_daily_generation, ensure_daily_room, get_today_key,
  apply_fragment_rule, verify_puzzle, verify_meta_final).

Minigames:
  A) Vault Frenzy        — tapping chaos + fake-outs
  B) Light Reactor       — precision timing + risk/reward
  C) Pressure Chamber    — real-time multitasking of valves

Expected external usage:
  - routes.py calls /escape/api/today, /escape/api/submit, /escape/api/finish
  - app/app.py may import schedule_daily_generation(app)  ← provided
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import string
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pytz

# ───────────────────────── Config ─────────────────────────
TIMEZONE = os.getenv("ESCAPE_TZ", "America/Boise")
_scheduler_started = False


# ───────────────────────── Secrets / RNG ─────────────────────────
def _app_secret() -> bytes:
    key = os.getenv("ESCAPE_SECRET") or os.getenv("SECRET_KEY") or "change-me"
    return key.encode("utf-8")


def _date_key(date: Optional[dt.date] = None) -> str:
    d = date or dt.date.today()
    return d.strftime("%Y-%m-%d")


def _seeded_rng(date: Optional[dt.date] = None, flavor: str = "") -> random.Random:
    seed_src = f"{_date_key(date)}::{flavor}::{os.getenv('ESCAPE_SEED_SALT','')}".encode()
    seed = int(hashlib.sha256(seed_src).hexdigest()[:16], 16)
    return random.Random(seed)


def _hmac_sign(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_app_secret(), payload, hashlib.sha256).hexdigest()


def _hmac_verify(data: Dict[str, Any], signature: str) -> bool:
    return hmac.compare_digest(_hmac_sign(data), signature)


# ───────────────────────── Back-compat helpers ─────────────────────────
def normalize_answer(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()


def get_today_key(tz: Optional[str] = None) -> str:
    now = dt.datetime.now(pytz.timezone(tz or TIMEZONE))
    return now.date().isoformat()


# ───────────────────────── Data Models (dataclasses) ─────────────────────────
@dataclass
class MiniConfig:
    type: str
    id: str
    seed: str
    time_cap_sec: int = 75


@dataclass
class VaultFrenzyConfig(MiniConfig):
    grid_rows: int = 3
    grid_cols: int = 4
    rounds: int = 3
    pops_per_round: List[int] = None
    decoys_per_round: List[int] = None
    allow_double_blink: bool = True
    allow_delay_blink: bool = True
    allow_wrong_color: bool = True


@dataclass
class LightReactorConfig(MiniConfig):
    rounds: int = 3
    orb_counts: List[int] = None
    speeds: List[float] = None
    target_spans_deg: List[float] = None
    reverse_rounds: List[bool] = None
    bonus_orb_prob: float = 0.35
    hit_tolerance_ms: int = 110


@dataclass
class PressureChamberConfig(MiniConfig):
    valves: int = 4
    rounds: int = 3
    rise_speeds: List[List[float]] = None
    overflow_threshold: float = 1.0
    reroll_speed_after_tap: bool = True


@dataclass
class DailyRoom:
    date: str
    theme: str
    minigames: List[Dict[str, Any]]
    server_private: Dict[str, Any]


# ───────────────────────── Generation ─────────────────────────
THEMES = [
    "Neon Archives", "Sapphire Vault", "Clockwork Corridor", "Lumen Lab",
    "Dusty Attic", "Void Observatory", "Arcade Bunker", "Crystal Caves",
]


def _pick_theme(rng: random.Random) -> str:
    return rng.choice(THEMES)


def _gen_vault_frenzy(rng: random.Random, ident="A") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = VaultFrenzyConfig(
        type="vault_frenzy",
        id=ident,
        seed=secrets.token_hex(8),
        grid_rows=3,
        grid_cols=4,
        rounds=3,
        pops_per_round=[rng.randint(3, 5) for _ in range(3)],
        decoys_per_round=[rng.randint(2, 4) for _ in range(3)],
        allow_double_blink=True,
        allow_delay_blink=True,
        allow_wrong_color=True,
        time_cap_sec=75,
    )
    public = asdict(cfg)
    return public, {"type": "vault_frenzy", "server_seed": rng.getrandbits(64)}


def _gen_light_reactor(rng: random.Random, ident="B") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = LightReactorConfig(
        type="light_reactor",
        id=ident,
        seed=secrets.token_hex(8),
        rounds=3,
        orb_counts=[1, 1, 2],
        speeds=[1.0, 1.25, 1.4],
        target_spans_deg=[40.0, 32.0, 28.0],
        reverse_rounds=[False, True, False],
        bonus_orb_prob=0.35,
        hit_tolerance_ms=110,
        time_cap_sec=75,
    )
    public = asdict(cfg)
    return public, {"type": "light_reactor", "server_seed": rng.getrandbits(64)}


def _gen_pressure_chamber(rng: random.Random, ident="C") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rounds = 3
    valves = 4
    rise_speeds = []
    for _ in range(rounds):
        rise_speeds.append([round(rng.uniform(0.25, 0.65), 2) for _ in range(valves)])
    cfg = PressureChamberConfig(
        type="pressure_chamber",
        id=ident,
        seed=secrets.token_hex(8),
        valves=valves,
        rounds=rounds,
        rise_speeds=rise_speeds,
        overflow_threshold=1.0,
        reroll_speed_after_tap=True,
        time_cap_sec=75,
    )
    public = asdict(cfg)
    return public, {"type": "pressure_chamber", "server_seed": rng.getrandbits(64)}


def generate_room(date: Optional[dt.date] = None) -> DailyRoom:
    rng = _seeded_rng(date, "room")
    theme = _pick_theme(rng)
    pubA, privA = _gen_vault_frenzy(rng, "A")
    pubB, privB = _gen_light_reactor(rng, "B")
    pubC, privC = _gen_pressure_chamber(rng, "C")
    server_private = {"minis": {"A": privA, "B": privB, "C": privC}}
    return DailyRoom(
        date=_date_key(date),
        theme=theme,
        minigames=[pubA, pubB, pubC],
        server_private=server_private,
    )


# ───────────────────────── Verification ─────────────────────────
class VerifyError(Exception):
    pass


def verify_vault_frenzy(server_seed: int, cfg: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    rng = random.Random(server_seed)
    rounds = int(cfg.get("rounds", 3))
    pops_per_round = cfg.get("pops_per_round", [3, 4, 4])

    total_ms = 0
    tr_rounds = transcript.get("rounds", [])
    if len(tr_rounds) != rounds:
        return (False, 0)
    for r in range(rounds):
        rd = tr_rounds[r]
        taps = rd.get("taps", [])
        elapsed = int(rd.get("elapsed_ms", 0))
        total_ms += max(0, elapsed)
        target_hits = pops_per_round[r]
        if len(taps) > 100:
            return (False, 0)
        if any(t.get("t", 0) < 0 for t in taps):
            return (False, 0)
        if len([t for t in taps if isinstance(t.get("idx"), int)]) < target_hits:
            return (False, 0)
    return (True, total_ms)


def verify_light_reactor(server_seed: int, cfg: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    rng = random.Random(server_seed)
    rounds = int(cfg.get("rounds", 3))
    speeds = cfg.get("speeds", [1.0, 1.25, 1.4])
    target_spans = cfg.get("target_spans_deg", [40.0, 32.0, 28.0])
    reverse = cfg.get("reverse_rounds", [False, True, False])

    stops = transcript.get("stops", [])
    elapsed = int(transcript.get("elapsed_ms", 0))

    if len(stops) < rounds:
        return (False, 0)
    # Assume ~1000 ms per round for verification
    for i in range(rounds):
        target_center = rng.uniform(0, 360.0)
        span = target_spans[i]
        t_i = int(stops[i].get("t", -1))
        if t_i < 0:
            return (False, 0)
        t_rel = (t_i % 1000) / 1000.0
        direction = -1.0 if reverse[i] else 1.0
        angle = (direction * 360.0 * speeds[i] * t_rel) % 360.0
        diff = abs((angle - target_center + 540.0) % 360.0 - 180.0)
        if diff > (span / 2.0) and diff > 5.0:
            return (False, 0)

    return (True, max(0, elapsed))


def verify_pressure_chamber(server_seed: int, cfg: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    rng = random.Random(server_seed)
    valves = int(cfg.get("valves", 4))
    rounds = int(cfg.get("rounds", 3))
    rise_speeds = cfg.get("rise_speeds", [[0.4]*valves]*rounds)
    threshold = float(cfg.get("overflow_threshold", 1.0))
    reroll = bool(cfg.get("reroll_speed_after_tap", True))

    actions = transcript.get("actions", [])
    elapsed = int(transcript.get("elapsed_ms", 0))

    idx = 0
    last_t = 0
    for r in range(rounds):
        press = [0.0] * valves
        speeds = list(rise_speeds[r])
        round_actions = []
        while idx < len(actions) and len(round_actions) < 20:
            t = int(actions[idx].get("t", -1))
            v = actions[idx].get("valve", -1)
            if t < last_t: break
            if t > (r+1)*1000: break
            round_actions.append((t, v))
            idx += 1

        cursor = r*1000
        for t, v in round_actions + [((r+1)*1000, None)]:
            dt_ms = max(0, t - cursor)
            for i in range(valves):
                press[i] += speeds[i] * (dt_ms / 1000.0)
                if press[i] >= threshold:
                    return (False, 0)
            cursor = t
            if v is None:
                break
            if not isinstance(v, int) or not (0 <= v < valves):
                return (False, 0)
            press[v] = 0.0
            if reroll:
                speeds[v] = round(rng.uniform(0.25, 0.65), 2)
        last_t = (r+1)*1000

    return (True, max(0, elapsed))


def verify_minigame(minigame_id: str, cfg: Dict[str, Any], server_private: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    minis = server_private.get("minis", {})
    if minigame_id not in minis:
        raise VerifyError("unknown minigame id")
    server_seed = int(minis[minigame_id].get("server_seed"))
    typ = cfg.get("type")
    if typ == "vault_frenzy":
        return verify_vault_frenzy(server_seed, cfg, transcript)
    if typ == "light_reactor":
        return verify_light_reactor(server_seed, cfg, transcript)
    if typ == "pressure_chamber":
        return verify_pressure_chamber(server_seed, cfg, transcript)
    raise VerifyError("unknown minigame type")


# ───────────────────────── Public payload ─────────────────────────
def room_public_payload(room: DailyRoom) -> Dict[str, Any]:
    return {
        "date": room.date,
        "theme": room.theme,
        "minigames": room.minigames,
        "signature": _hmac_sign({"date": room.date, "count": len(room.minigames)}),
    }


def fragment_for(mini_id: str) -> str:
    return {"A": "◆", "B": "◎", "C": "✶"}.get(mini_id, "?")


def assemble_final_code(fragments: List[str]) -> str:
    return "".join(fragments)


# ───────────────────────── Persistence + Scheduler (compat) ─────────────────────────
def _get_db_and_models():
    from app.extensions import db
    from .models import EscapeRoom
    return db, EscapeRoom


def ensure_daily_room(date_key: Optional[str] = None, force_regen: bool = False):
    db, EscapeRoom = _get_db_and_models()
    if date_key is None:
        date_key = get_today_key()
    row = db.session.query(EscapeRoom).filter_by(date=date_key).first()
    if row and not force_regen:
        return row
    dr = generate_room(dt.datetime.strptime(date_key, "%Y-%m-%d").date())
    if row:
        row.theme = dr.theme
        row.minigames_json = dr.minigames
        row.server_private_json = dr.server_private
        db.session.add(row); db.session.commit(); return row
    else:
        row = EscapeRoom(
            date=dr.date, theme=dr.theme,
            minigames_json=dr.minigames, server_private_json=dr.server_private
        )
        db.session.add(row); db.session.commit(); return row


def schedule_daily_generation(app) -> None:
    """Start a background scheduler to pre-generate the daily room (00:05 local)."""
    global _scheduler_started
    if _scheduler_started:
        try: app.logger.info("[escape] scheduler already started; skipping.")
        except Exception: pass
        return
    _scheduler_started = True
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as e:
        try: app.logger.warning(f"[escape] APScheduler not available: {e}")
        except Exception: pass
        return

    tz = pytz.timezone(TIMEZONE)
    scheduler = BackgroundScheduler(timezone=tz)

    def job():
        with app.app_context():
            date_key = get_today_key()
            try:
                ensure_daily_room(date_key)
                app.logger.info(f"[escape] Daily room generated for {date_key}")
            except Exception as e:
                app.logger.error(f"[escape] Daily generation failed for {date_key}: {e}")

    scheduler.add_job(job, "cron", hour=0, minute=5, id="escape_daily_gen", replace_existing=True)
    scheduler.start()
    try: app.logger.info("[escape] scheduler started (daily 00:05 local).")
    except Exception: pass


# ───────────────────────── Legacy puzzle-API compatibility ─────────────────────────
def apply_fragment_rule(answer: str, rule: str) -> str:
    """
    Keeps compatibility with older meta flows.
    Supports: CONST:<TEXT>, FIRST2, FIRST3, LAST2, LAST3, CAESAR:+/-K;FIRST2/3/LAST2/3, IDX:a,b,c
    """
    ans = normalize_answer(answer)
    rule = (rule or "").strip().upper()

    if rule.startswith("CONST:"):
        return normalize_answer(rule.split("CONST:",1)[1])[:4]

    if rule == "FIRST2":  return ans[:2]
    if rule == "FIRST3":  return ans[:3]
    if rule == "LAST2":   return ans[-2:]
    if rule == "LAST3":   return ans[-3:]

    import re as _re
    m = _re.match(r"CAESAR:\+?(-?\d+);(FIRST2|FIRST3|LAST2|LAST3)$", rule)
    if m:
        k = int(m.group(1)) % 26
        A = string.ascii_uppercase
        shifted = "".join(A[(A.index(ch)+k)%26] if ch in A else ch for ch in ans)
        sub = m.group(2)
        return apply_fragment_rule(shifted, sub)

    m = _re.match(r"IDX:([0-9,]+)$", rule)
    if m:
        idxs = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
        out = "".join(ans[i] for i in idxs if 0 <= i < len(ans))
        return out[:4]

    return ans[:4]


def verify_puzzle(room_json: Dict[str, Any], puzzle_id: str, answer: str) -> bool:
    """
    Minimal backward-compat. If legacy 'puzzles' exist, check solution.answer.
    This will return False for our new minigames (which use /api/submit instead).
    """
    puzzles = (room_json or {}).get("puzzles") or []
    for p in puzzles:
        if not isinstance(p, dict): continue
        if p.get("id") != puzzle_id: continue
        sol = (p.get("solution") or {}).get("answer") or p.get("answer")
        if sol is None: return False
        return normalize_answer(str(sol)) == normalize_answer(str(answer or ""))
    return False


def verify_meta_final(room_json: Dict[str, Any], submitted: str) -> bool:
    final = (room_json or {}).get("final") or {}
    sol = (final.get("solution") or {}).get("answer") or final.get("answer")
    if sol is None: return False
    return normalize_answer(str(sol)) == normalize_answer(str(submitted or ""))
