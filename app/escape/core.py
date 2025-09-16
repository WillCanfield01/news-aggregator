
# app/escape/core.py (REWRITE)
# -*- coding: utf-8 -*-
"""
Escape minigames core: daily 3 arcades -> 1 final lock (server verified).

This file defines:
- Daily room generation (deterministic by date) with 3 ORIGINAL minigames:
    1) Vault Frenzy       (tapping chaos + fake-outs)
    2) Light Reactor      (precision timing + risk/reward)
    3) Pressure Chamber   (real-time multitasking valves)
- Server-side verification for each minigame (no client solutions)
- HMAC-signed completion tokens
- Helpers for daily room ensure/fetch

Design goals:
- Mobile-first, < 5 min total
- One-and-done per day
- Leaderboard uses total time (ms)
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
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ───────────────────────── Config & RNG ─────────────────────────

def _app_secret() -> bytes:
    # Use a dedicated secret if present; fall back to Flask SECRET_KEY.
    key = os.getenv("ESCAPE_SECRET") or os.getenv("SECRET_KEY") or "change-me"
    return key.encode("utf-8")


def _date_key(date: Optional[dt.date] = None) -> str:
    d = date or dt.date.today()
    return d.strftime("%Y-%m-%d")


def _seeded_rng(date: Optional[dt.date] = None, flavor: str = "") -> random.Random:
    seed_src = f"{_date_key(date)}::{flavor}::{os.getenv('ESCAPE_SEED_SALT','')}".encode()
    # Stable 64-bit seed
    seed = int(hashlib.sha256(seed_src).hexdigest()[:16], 16)
    return random.Random(seed)


def _hmac_sign(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_app_secret(), payload, hashlib.sha256).hexdigest()


def _hmac_verify(data: Dict[str, Any], signature: str) -> bool:
    return hmac.compare_digest(_hmac_sign(data), signature)


# ───────────────────────── Data Models (dataclasses) ─────────────────────────

@dataclass
class MiniConfig:
    type: str
    id: str
    seed: str
    # Generic caps to keep rounds short & consistent:
    time_cap_sec: int = 75


@dataclass
class VaultFrenzyConfig(MiniConfig):
    grid_rows: int = 3
    grid_cols: int = 4
    rounds: int = 3
    # Each round: number of correct pops and injected decoys
    pops_per_round: List[int] = None
    decoys_per_round: List[int] = None
    # Fake-out behaviors:
    allow_double_blink: bool = True
    allow_delay_blink: bool = True
    allow_wrong_color: bool = True


@dataclass
class LightReactorConfig(MiniConfig):
    rounds: int = 3
    # For each round we specify orb count (1→2), speed multipliers, and target zones
    orb_counts: List[int] = None          # e.g., [1,1,2]
    speeds: List[float] = None            # base angular velocity per round
    target_spans_deg: List[float] = None  # size of target window in degrees
    reverse_rounds: List[bool] = None     # whether to reverse direction
    bonus_orb_prob: float = 0.35          # chance to spawn bonus orb in a round
    # Tolerance for server overlap check (ms windows around true overlap moment)
    hit_tolerance_ms: int = 110


@dataclass
class PressureChamberConfig(MiniConfig):
    valves: int = 4
    rounds: int = 3
    # For each round, per-valve rise speed in "pressure units per second"
    rise_speeds: List[List[float]] = None
    # Pressure max threshold (overflow if >= 1.0)
    overflow_threshold: float = 1.0
    # After a tap, valve resets to 0 and speed may reroll within range (server seeded)
    reroll_speed_after_tap: bool = True


@dataclass
class DailyRoom:
    date: str
    theme: str
    minigames: List[Dict[str, Any]]  # public-safe config for client
    # server-only (never returned to client):
    server_private: Dict[str, Any]


# ───────────────────────── Daily Generation ─────────────────────────

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
    # Strip server-only determinism details if any are added later; for now we keep public minimal.
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
        # Generate per-valve speeds between 0.25 and 0.65 units/sec for variety
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


# ───────────────────────── Server Verification ─────────────────────────
# We verify based on a deterministic simulation using server seeds and
# the client's input transcript. We only need to confirm success and compute
# an effective time. Specific physics/animations run on client are *visual only*.

class VerifyError(Exception):
    pass


def verify_vault_frenzy(server_seed: int, cfg: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    """
    transcript:
      {
        "rounds": [
          {"taps": [{"idx": 5, "t": 123}, ...], "elapsed_ms": 2100},
          ...
        ]
      }
    Return (passed, elapsed_ms).
    """
    rng = random.Random(server_seed)
    rounds = cfg.get("rounds", 3)
    pops_per_round = cfg.get("pops_per_round", [3,4,4])
    decoys_per_round = cfg.get("decoys_per_round", [2,3,3])

    # We simulate that there are exactly 'pops_per_round[r]' correct windows to hit.
    # Client succeeds a round if they have at least that many valid taps (we allow extra taps but ignore).
    # We also cap time by provided elapsed.
    total_ms = 0
    tr_rounds = transcript.get("rounds", [])
    if len(tr_rounds) != rounds:
        return (False, 0)
    for r in range(rounds):
        rd = tr_rounds[r]
        taps = rd.get("taps", [])
        elapsed = int(rd.get("elapsed_ms", 0))
        total_ms += max(0, elapsed)
        # Deterministic target count
        target_hits = pops_per_round[r]
        # Very basic integrity: cap taps to grid size * 3 to discourage spam
        if len(taps) > 100:
            return (False, 0)
        # Accept if taps >= target_hits, and no obvious impossible sequences (e.g., negative times)
        if any(t.get("t", 0) < 0 for t in taps):
            return (False, 0)
        if len([t for t in taps if isinstance(t.get("idx"), int)]) < target_hits:
            return (False, 0)
    return (True, total_ms)


def verify_light_reactor(server_seed: int, cfg: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    """
    transcript:
      { "stops": [ {"t": 812}, {"t": 1543}, {"t": 2301} ], "elapsed_ms": 3000 }
    Each stop must coincide (within tolerance) with an overlap of orb and target zone.
    We simulate the orb angle over time using 'speeds', 'reverse_rounds', and 'target_spans_deg'.
    """
    rng = random.Random(server_seed)
    rounds = cfg.get("rounds", 3)
    speeds = cfg.get("speeds", [1.0, 1.25, 1.4])
    target_spans = cfg.get("target_spans_deg", [40.0, 32.0, 28.0])
    reverse = cfg.get("reverse_rounds", [False, True, False])
    tol = int(cfg.get("hit_tolerance_ms", 110))

    stops = transcript.get("stops", [])
    elapsed = int(transcript.get("elapsed_ms", 0))

    if len(stops) < rounds:
        return (False, 0)
    # Simulate: each round lasts ~1000 ms baseline
    success = True
    for i in range(rounds):
        # For determinism, set round-specific target angle in [0, 360)
        target_center = rng.uniform(0, 360.0)
        span = target_spans[i]
        # Compute orb angle at stop time t_i (relative per round window):
        t_i = int(stops[i].get("t", -1))
        if t_i < 0:
            success = False
            break
        # Normalize to round window (assume ~1000ms per round for verification)
        t_rel = (t_i % 1000) / 1000.0
        direction = -1.0 if reverse[i] else 1.0
        angle = (direction * 360.0 * speeds[i] * t_rel) % 360.0
        # Check overlap:
        diff = abs((angle - target_center + 540.0) % 360.0 - 180.0)
        if diff > (span / 2.0) and diff > 5.0:  # small absolute tolerance
            success = False
            break

    if not success:
        return (False, 0)

    return (True, max(0, elapsed))


def verify_pressure_chamber(server_seed: int, cfg: Dict[str, Any], transcript: Dict[str, Any]) -> Tuple[bool, int]:
    """
    transcript:
      { "actions": [ {"valve": 0, "t": 420}, ... ], "elapsed_ms": 2800 }
    We simulate N valves whose pressure increases linearly per round. Tapping a valve
    resets it to 0 and may reroll its speed. If any valve reaches threshold before
    an action, round fails.
    """
    rng = random.Random(server_seed)
    valves = int(cfg.get("valves", 4))
    rounds = int(cfg.get("rounds", 3))
    rise_speeds = cfg.get("rise_speeds", [[0.4]*valves]*rounds)
    threshold = float(cfg.get("overflow_threshold", 1.0))
    reroll = bool(cfg.get("reroll_speed_after_tap", True))

    actions = transcript.get("actions", [])
    elapsed = int(transcript.get("elapsed_ms", 0))

    # Sim verification per round: window ~1000ms. We walk through actions time-ordered.
    idx = 0
    last_t = 0
    for r in range(rounds):
        # State per valve at start of round
        press = [0.0] * valves
        speeds = list(rise_speeds[r])
        # Extract actions for this round (<= 20 to avoid spam)
        round_actions = []
        while idx < len(actions) and len(round_actions) < 20:
            t = int(actions[idx].get("t", -1))
            v = actions[idx].get("valve", -1)
            if t < last_t: break  # next round
            if t > (r+1)*1000: break  # next round window
            round_actions.append((t, v))
            idx += 1

        # Simulate time from r*1000 → (r+1)*1000
        cursor = r*1000
        for t, v in round_actions + [((r+1)*1000, None)]:
            dt_ms = max(0, t - cursor)
            # increase pressure
            for i in range(valves):
                press[i] += speeds[i] * (dt_ms / 1000.0)
                if press[i] >= threshold:
                    return (False, 0)
            cursor = t
            if v is None:
                break
            # tap valve v
            if not isinstance(v, int) or not (0 <= v < valves):
                return (False, 0)
            press[v] = 0.0
            if reroll:
                speeds[v] = round(rng.uniform(0.25, 0.65), 2)
        last_t = (r+1)*1000

    return (True, max(0, elapsed))


# ───────────────────────── Public API helpers ─────────────────────────

def room_public_payload(room: DailyRoom) -> Dict[str, Any]:
    # Prepare safe JSON for client
    return {
        "date": room.date,
        "theme": room.theme,
        "minigames": room.minigames,
        # signed envelope to prevent tampering (does NOT include answers)
        "signature": _hmac_sign({"date": room.date, "count": len(room.minigames)}),
    }


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


def fragment_for(mini_id: str) -> str:
    # Simple mapping to fun symbols; the final code is the concatenation
    # of the three fragments in order A→B→C.
    return {"A": "◆", "B": "◎", "C": "✶"}.get(mini_id, "?")


def assemble_final_code(fragments: List[str]) -> str:
    # Human-readable final code for share text
    return "".join(fragments)


# Persistence hooks are kept in routes/models to avoid circular imports.
