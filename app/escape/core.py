# app/escape_core.py
# Clean, feature-complete core for Daily Escape with ONLY 3 minis:
# ◆ Vault Frenzy • ◎ Phantom Doors • ✶ Pressure Chamber
# Provides: /escape/api/today, /escape/api/submit, /escape/api/finish, /escape/leaderboard

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, re, random, time, hashlib, math

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default

DAILY_SECRET = env("DAILY_SECRET", "fallback-secret")
LEADERBOARD_SIZE = int(env("ESCAPE_LEADERBOARD_SIZE", "100"))

def daily_seed(date_key: str, secret: Optional[str] = None) -> int:
    """Stable per-day seed from YYYY-MM-DD + secret."""
    secret = (secret or DAILY_SECRET).encode("utf-8")
    dk = (date_key or "").encode("utf-8")
    h = hashlib.sha256(secret + b"|" + dk).hexdigest()
    return int(h[:8], 16) & 0x7fffffff

def rng_from_seed(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r

def normalize_answer(s: Any) -> str:
    """Uppercase, strip non-alphanumerics except comma/hyphen for uniform checks."""
    if s is None:
        return ""
    s = str(s)
    return re.sub(r"[^A-Za-z0-9,\-]+", "", s).upper()

def now_ms() -> int:
    return int(time.time() * 1000)

# --------------------------------------------------------------------
# Data Model
# --------------------------------------------------------------------

@dataclass
class Mini:
    id: str
    archetype: str
    mechanic: str
    prompt: str
    answer_format: Dict[str, Any]
    solution: Dict[str, Any]
    hints: List[str]
    decoys: List[str]
    paraphrases: List[str]
    ui_spec: Dict[str, Any]

def mini_to_dict(m: Mini) -> Dict[str, Any]:
    d = asdict(m)
    d["type"] = d.get("archetype", "mini")
    d["puzzle_id"] = d.get("id")
    return d

# --------------------------------------------------------------------
# Mini Generators (deterministic from rng)
# --------------------------------------------------------------------

def gen_vault_frenzy(rng: random.Random, pid: str, theme: str = "") -> Mini:
    rows, cols = 4, 4
    total = rows * cols
    true_count, decoy_count = 6, 2

    choices = list(range(total))
    rng.shuffle(choices)
    true_indices = sorted(choices[:true_count])
    decoy_indices = sorted(choices[true_count:true_count + decoy_count])

    prompt = (
        f"{theme or 'The Salt Vault'}: Watch the vault nodes. "
        "Tap the ones that flash green, in order. Ignore the decoys."
    )

    return Mini(
        id=pid,
        archetype="mini",
        mechanic="vault_frenzy",
        prompt=prompt,
        answer_format={"pattern": r"^\d+(?:,\d+){5}$"},
        solution={"answer": ",".join(str(i) for i in true_indices)},
        hints=[
            "You will see 6 true flashes. Decoys may double-blink.",
            "Order matters—tap in the same order they turned green."
        ],
        decoys=[],
        paraphrases=[],
        ui_spec={
            "kind": "vault_frenzy",
            "grid_rows": rows,
            "grid_cols": cols,
            "tempo_ms": 540,
            "true_indices": true_indices,
            "decoy_indices": decoy_indices,
            "rules": {"double_blink_decoys": True},
        },
    )

def gen_phantom_doors(rng: random.Random, pid: str, theme: str = "") -> Mini:
    # 6 glyphs from a larger set
    glyphs = ["◆", "◇", "◼︎", "△", "✦", "✪", "✷", "✚", "✖︎", "✿", "●", "■"]
    rng.shuffle(glyphs)
    symbols = glyphs[:6]

    rounds = 3
    # Target sequence expressed in SYMBOL space (0..5)
    sequence = [rng.randrange(6) for _ in range(rounds)]

    # Per-round mapping position -> symbol index
    shuffles: List[List[int]] = []
    phantoms: List[int] = []
    for _ in range(rounds):
        perm = list(range(6))
        rng.shuffle(perm)
        shuffles.append(perm)
        phantoms.append(rng.randrange(6))  # one phantom position per round

    prompt = (
        f"{theme or 'The Cerulean Observatory'}: Memorize the symbol sequence. "
        "Doors reshuffle each round. Avoid phantom doors (they fade)!"
    )

    return Mini(
        id=pid,
        archetype="mini",
        mechanic="phantom_doors",
        prompt=prompt,
        answer_format={"pattern": r"^\d+(?:,\d+){2}$"},  # three integers
        # We store solution in SYMBOL indices; client may submit symbol or position
        solution={"answer": ",".join(str(i) for i in sequence)},
        hints=[
            "Look at the sequence preview, then pick the matching symbol each round.",
            "Phantoms fade if tapped—skip them."
        ],
        decoys=[],
        paraphrases=[],
        ui_spec={
            "kind": "phantom_doors",
            "symbols": symbols,
            "rounds": rounds,
            "sequence": sequence,     # explicit target in symbol space
            "shuffles": shuffles,     # per-round pos->symbol mapping
            "phantoms": phantoms,     # position index that is phantom
            "tempo_ms": 600,
        },
    )

def gen_pressure_chamber(rng: random.Random, pid: str, theme: str = "") -> Mini:
    n = rng.randrange(3, 6)  # 3–5 valves
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:n]
    valves = []
    for i in range(n):
        valves.append({
            "label": labels[i],
            "base": round(rng.uniform(0.16, 0.32), 3),
            "var": round(rng.uniform(0.05, 0.12), 3),
            "reset_ms": rng.randrange(700, 1200),
            "phase": round(rng.uniform(0.0, 1.0), 2),
        })

    # Solution is informational; client decides win, server records finish
    order = [rng.choice(labels) for _ in range(n * 2)]
    prompt = (
        f"{theme or 'Deepline Chamber'}: Keep all gauges below the red line until the timer ends. "
        "Tap a valve to dump pressure; each climbs at a different rate."
    )

    return Mini(
        id=pid,
        archetype="mini",
        mechanic="pressure_chamber",
        prompt=prompt,
        answer_format={"pattern": r"^[A-Z](,[A-Z])+$"},
        solution={"answer": ",".join(order)},
        hints=[f"{n} valves—watch the fastest riser.", "A tap drops pressure; it climbs again quickly."],
        decoys=[],
        paraphrases=[],
        ui_spec={
            "kind": "pressure_chamber",
            "seed": rng.randrange(2**31),
            "valves": valves,
            "threshold": 1.0,
            "survive_ms": 10000,   # front-end reads this
        },
    )

# --------------------------------------------------------------------
# Assembly & Verification
# --------------------------------------------------------------------

def build_minis(date_key: str, theme: str = "Daily Escape") -> List[Dict[str, Any]]:
    rng = rng_from_seed(daily_seed(date_key))
    return [
        mini_to_dict(gen_vault_frenzy(rng, "vault_frenzy", theme)),
        mini_to_dict(gen_phantom_doors(rng, "phantom_doors", theme)),
        mini_to_dict(gen_pressure_chamber(rng, "pressure_chamber", theme)),
    ]

def verify_vault(mini: Dict[str, Any], answer: str) -> bool:
    want = normalize_answer((mini.get("solution") or {}).get("answer", ""))
    got = normalize_answer(answer)
    return want == got

def verify_doors(mini: Dict[str, Any], answer: str) -> bool:
    # Accept either symbol-sequence (stored) or position-sequence translated to symbols
    want_raw = (mini.get("solution") or {}).get("answer", "")
    want = [int(x) for x in normalize_answer(want_raw).split(",") if x != ""]
    got = [int(x) for x in normalize_answer(answer).split(",") if x != ""]
    if got == want:
        return True
    ui = mini.get("ui_spec") or {}
    shuffles = ui.get("shuffles") or []
    if len(got) == len(want) and len(shuffles) >= len(got):
        try:
            # positions -> symbols via per-round pos->symbol
            as_sym = [shuffles[i][got[i]] for i in range(len(got))]
            return as_sym == want
        except Exception:
            return False
    return False

def verify_answer(mini: Dict[str, Any], answer: str) -> bool:
    mech = (mini.get("mechanic") or "").lower()
    if mech == "vault_frenzy":
        return verify_vault(mini, answer)
    if mech == "phantom_doors":
        return verify_doors(mini, answer)
    if mech == "pressure_chamber":
        # Client decides win; treat non-empty as OK to store a local completion
        return bool(normalize_answer(answer))
    return False

# --------------------------------------------------------------------
# In-memory Store (swap to Redis/Postgres in production)
# --------------------------------------------------------------------

class Store:
    def __init__(self):
        # completions[(date_key, client_id)] = {puzzle_id: answer}
        self.completions: Dict[Tuple[str, str], Dict[str, str]] = {}
        # finishes[date_key] = list of {client_id, time_ms, success, started_ms, finished_ms}
        self.finishes: Dict[str, List[Dict[str, Any]]] = {}

    def mark_correct(self, date_key: str, client_id: str, puzzle_id: str, answer: str):
        key = (date_key, client_id)
        m = self.completions.get(key) or {}
        m[puzzle_id] = str(answer)
        self.completions[key] = m

    def get_cleared_count(self, date_key: str, client_id: str) -> int:
        return len(self.completions.get((date_key, client_id)) or {})

    def add_finish(self, date_key: str, client_id: str, started_ms: int, time_ms: int, success: bool):
        L = self.finishes.get(date_key) or []
        L.append({
            "client_id": client_id,
            "time_ms": int(time_ms),
            "success": bool(success),
            "started_ms": int(started_ms),
            "finished_ms": now_ms(),
        })
        # keep small
        L.sort(key=lambda x: (not x["success"], x["time_ms"]))
        self.finishes[date_key] = L[:LEADERBOARD_SIZE]

    def top(self, date_key: str) -> List[Dict[str, Any]]:
        return list(self.finishes.get(date_key) or [])

STORE = Store()

# --------------------------------------------------------------------
# FastAPI App + Routes
# --------------------------------------------------------------------

router = APIRouter(prefix="/escape/api", tags=["escape"])

def client_id_from(request: Request) -> str:
    # Prefer a header you already set in your app (user/session). Fallback to IP.
    return (
        request.headers.get("X-Player-Id")
        or request.headers.get("X-Session-Id")
        or request.client.host
        or "anon"
    )

@router.get("/today")
async def get_today(format: Optional[str] = "minis", date: Optional[str] = ""):
    date_key = date or time.strftime("%Y-%m-%d", time.gmtime())
    if format != "minis":
        raise HTTPException(400, detail="only format=minis is supported in this core")
    minis = build_minis(date_key)
    return {"theme": "Daily Escape", "minigames": minis, "date_key": date_key}

@router.post("/submit")
async def post_submit(req: Request):
    body = await req.json()
    date_key = body.get("date_key") or body.get("date") or time.strftime("%Y-%m-%d", time.gmtime())
    puzzle_id = (body.get("puzzle_id") or body.get("id") or "").strip()
    answer = body.get("answer") or ""

    if not puzzle_id:
        raise HTTPException(400, detail="puzzle_id required")

    # regenerate minis deterministically for server-side check
    minis = {m["puzzle_id"]: m for m in build_minis(date_key)}
    mini = minis.get(puzzle_id)
    if not mini:
        raise HTTPException(404, detail="unknown puzzle_id")

    correct = verify_answer(mini, answer)
    if correct:
        STORE.mark_correct(date_key, client_id_from(req), puzzle_id, str(answer))

    return JSONResponse({"ok": True, "correct": bool(correct)})

@router.post("/finish")
async def post_finish(req: Request):
    """
    Body:
      - date_key
      - started_ms (client-start)
      - success (bool)
      - meta: { minis_only: bool, enforce_final: bool }  # ignored here
    Returns { ok, success, time_ms }
    """
    body = await req.json()
    date_key = body.get("date_key") or time.strftime("%Y-%m-%d", time.gmtime())
    success = bool(body.get("success", True))
    started_ms = int(body.get("started_ms") or now_ms())

    # Authoritative server time
    finished_ms = now_ms()
    time_ms = max(0, finished_ms - started_ms)

    # Record result (store even if not all cleared; sorted by success/time)
    STORE.add_finish(date_key, client_id_from(req), started_ms, time_ms, success)

    return JSONResponse({"ok": True, "success": success, "time_ms": time_ms})

# Public leaderboard helper (JSON)
lb_router = APIRouter(prefix="/escape", tags=["escape"])

@lb_router.get("/leaderboard")
async def get_leaderboard(date: Optional[str] = ""):
    date_key = date or time.strftime("%Y-%m-%d", time.gmtime())
    rows = STORE.top(date_key)
    # tiny share block summary (same format your front-end prints)
    best = next((r for r in rows if r.get("success")), None)
    share = None
    if best:
        mm = best["time_ms"] // 60000
        ss = (best["time_ms"] % 60000) // 1000
        tenths = (best["time_ms"] % 1000) // 100
        share = f"Daily Escape {date_key}\n(Top) {mm}:{str(ss).zfill(2)}.{tenths}"
    return {"date_key": date_key, "top": rows, "share": share}

# Application
def create_app() -> FastAPI:
    app = FastAPI(title="Daily Escape Core (3 minis)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(lb_router)
    return app

# For "uvicorn app.escape_core:app --reload"
app = create_app()
