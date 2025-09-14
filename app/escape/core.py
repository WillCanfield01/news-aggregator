# app/escape/core.py
# -*- coding: utf-8 -*-
"""
Trailroom core: daily 3 rooms -> 1 final lock.

- OpenAI generates atmosphere + puzzles (strict JSON; sanitized).
- Three routes per room; BOTH yield the same fragment (server-enforced).
- Server verifies all answers; client never receives solutions.
- Flat `puzzles` list kept for backward-compat with existing routes.
"""

from __future__ import annotations

import os, hmac, hashlib, random, string, re, json
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytz
from flask import current_app

# ─────────────────────────────────────────────────────────────────────────────
# Avoid circular imports: import db/models only at call sites.
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_and_models():
    from app.extensions import db
    from .models import EscapeRoom
    return db, EscapeRoom

# ───────────────────────── Config ─────────────────────────

RECENT_WINDOW_DAYS = 60
ANSWER_COOLDOWN_DAYS = 14
MAX_GEN_ATTEMPTS = 3
DEFAULT_DIFFICULTY = "medium"
TIMEZONE = os.getenv("ESCAPE_TZ", "America/Boise")
SUPPLIES_START_DEFAULT = 3

COMMON_STOCK_ANSWERS = {
    "piano","keyboard","silence","time","shadow","map","echo","fire","ice",
    "darkness","light","egg","door","wind","river"
}

ANAGRAM_WORDS = [
    "lantern","saffron","garnet","amplify","banquet","topaz","onyx","galaxy",
    "harbor","cobalt","jasper","velvet","sepia","orchid","monsoon","rift",
    "cipher","lilac","ember","quartz","krypton","zephyr","coral","indigo","scarlet"
]

THEMES = [
    ("The Clockmaker’s Loft","Dusty gears tick as you wake beneath copper skylights."),
    ("Signal in the Sublevel","A faint hum from heavy conduits thrums through the floor."),
    ("Archive of Ash","Charred shelves lean, cradling sealed folios that smell of smoke."),
    ("Night at the Conservatory","Moonlight fractures through greenhouse panes onto damp stone."),
    ("The Salt Vault","Clink… clink… droplets echo in a chalk-white storehouse."),
    ("Ferry to Nowhere","River fog swallows the terminal as lanterns pulse and fade."),
    ("Radio Silence","A dead station blinks a lone cursor on a green-glass screen."),
]

FINAL_CODE_MIN_LEN = 4
FINAL_CODE_MAX_LEN = 12

# Allowed puzzle archetypes (LLM and offline)
ALLOWED_TYPES = {"mini", "acrostic", "tapcode", "pathcode", "anagram", "caesar", "vigenere", "numeric_lock"}
ALLOWED_MECHANICS = {"multiple_choice", "sequence_input", "grid_input", "text_input"}
# Shared control chips for sequence minis
SEQ_TOKENS = ["tap","hold","left","right","up","down","rotate_left","rotate_right"]

NUM_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve"
}

def _looks_trivial_multiple_choice(p: Dict[str, Any]) -> bool:
    if (p.get("type") != "mini") or (p.get("mechanic") != "multiple_choice"):
        return False

    ui = p.get("ui_spec") or {}
    opts = ui.get("options") or []
    prompt = (p.get("prompt") or "").lower()

    # Too few options
    if len(opts) < 4:
        return True

    # “name that number”-style wording (existing)
    if re.search(r"(which|what).*(name|names|word).*(number|digit)", prompt):
        return True

    # All options are simple number words or digits (existing)
    def _is_num_token(x: Any) -> bool:
        s = str(x).strip().lower()
        return bool(re.fullmatch(r"\d{1,2}", s) or s in NUM_WORDS)
    if opts and all(_is_num_token(o) for o in opts):
        return True

    # NEW: generic shape/pattern guessing (“which pattern/shape/figure?”) with stocky options
    generic = {"loop","spiral","echo","pulse","wave","circle","ring","arc","line"}
    if re.search(r"\b(which|what)\b.*\b(pattern|shape|figure|symbol)\b", prompt):
        hits = sum(1 for o in opts if str(o).strip().lower() in generic)
        if hits >= max(2, len(opts) - 2):   # most options are generic shapes
            return True

    # NEW: any option is a known cliché/stock riddle answer → trivial
    if any(str(o).strip().lower() in {s.lower() for s in COMMON_STOCK_ANSWERS} for o in opts):
        return True

    # NEW: enforce “≥2 clues” heuristic for MC — require at least two bullet/numbered lines
    clue_lines = re.findall(r"(?m)^\s*(?:[-•]|[0-9]+\)|[0-9]+\.)", prompt)
    if re.search(r"\b(which|what)\b", prompt) and len(clue_lines) < 2:
        return True

    return False

def _make_sequence_mini(rng: random.Random, pid: str, theme: str) -> Dict[str, Any]:
    tokens = SEQ_TOKENS[:]
    k = rng.randrange(7, 10)  # 7–9 steps (slightly harder)
    seq = [rng.choice(tokens) for _ in range(k)]
    return {
        "id": pid,
        "type": "mini",
        "mechanic": "sequence_input",
        "ui_spec": {"sequence": tokens},
        "prompt": (
            f"Clockwork in the {theme or 'room'} clicks a repeating pattern. "
            "Reproduce the sequence using the chips.\n"
            "Controls:\n"
            "- tap = press the brass button\n"
            "- hold = keep the button pressed\n"
            "- left/right/up/down = nudge the lever\n"
            "- rotate_left/right = turn the crank counter/clockwise"
        ),
        "answer_format": {"pattern": r"^[A-Za-z0-9,\-]{5,24}$"},
        "solution": {"answer": ",".join(seq)},
        "hints": [f"Sequence length: {k}.", "Use only the chips above; order matters."]
    }

# ───────────────────────── Utilities ─────────────────────────

def _is_numeric(s: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(s or "")))

def _default_pattern_for_answer(answer: str) -> str:
    # Compact, anchored patterns; allow underscores for tokens like rotate_left
    a = str(answer or "")
    if _is_numeric(a): return r"^\d{1,12}$"
    if re.fullmatch(r"[A-Za-z]+", a): return r"^[A-Za-z]{2,16}$"
    # allow short tokens with commas/dashes/underscores (e.g., sequences)
    return r"^[A-Za-z0-9,_\-]{1,64}$"

def daily_seed(date_key: str, secret: str) -> int:
    digest = hmac.new(secret.encode("utf-8"), date_key.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big", signed=False)

def rng_from_seed(seed: int) -> random.Random:
    r = random.Random(); r.seed(seed); return r

def get_today_key(tz: Optional[str] = None) -> str:
    now = dt.datetime.now(pytz.timezone(tz or TIMEZONE))
    return now.date().isoformat()

def normalize_answer(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()

def _as_dict_list(seq) -> List[Dict[str, Any]]:
    """Filter any sequence down to dicts only."""
    return [x for x in (seq or []) if isinstance(x, dict)]

def _shingles(text: str, k: int = 5) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < k: return {t}
    return {t[i:i+k] for i in range(len(t)-k+1)}

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return 0.0 if union == 0 else inter/union

def recent_rooms(window_days: int = RECENT_WINDOW_DAYS) -> List[Any]:
    db, EscapeRoom = _get_db_and_models()
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=window_days)
    return (db.session.query(EscapeRoom)
            .filter(EscapeRoom.created_at >= cutoff)
            .order_by(EscapeRoom.created_at.desc())
            .limit(120).all())

def is_too_similar_to_recent(room_json: Dict[str, Any],
                             window_days: int = RECENT_WINDOW_DAYS,
                             sim_threshold: float = 0.35) -> bool:
    text = f"{room_json.get('title','')} {room_json.get('intro','')}"
    for rm in _as_dict_list(room_json.get("trail", {}).get("rooms")):
        text += " " + rm.get("title","") + " " + rm.get("text","")
        for r in _as_dict_list(rm.get("routes")):
            p = r.get("puzzle") if isinstance(r.get("puzzle"), dict) else {}
            text += " " + (p.get("prompt") or "")
    S = _shingles(text, k=7)
    for r in recent_rooms(window_days):
        blob = r.json_blob or {}
        t2 = f"{blob.get('title','')} {blob.get('intro','')}"
        for rm in _as_dict_list(blob.get("trail", {}).get("rooms")):
            t2 += " " + rm.get("title","") + " " + rm.get("text","")
            for rr in _as_dict_list(rm.get("routes")):
                pp = rr.get("puzzle") if isinstance(rr.get("puzzle"), dict) else {}
                t2 += " " + (pp.get("prompt") or "")
        if jaccard(S, _shingles(t2, k=7)) >= sim_threshold:
            return True
    return False

def answer_recently_used(answer: str, cooldown_days: int = ANSWER_COOLDOWN_DAYS) -> bool:
    db, EscapeRoom = _get_db_and_models()
    ans_norm = normalize_answer(answer)
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=cooldown_days)
    rs = db.session.query(EscapeRoom).filter(EscapeRoom.created_at >= cutoff).all()
    for r in rs:
        blob = r.json_blob or {}
        for p in _as_dict_list(blob.get("puzzles")):
            sol = (p.get("solution") or {}).get("answer")
            if sol and normalize_answer(sol) == ans_norm:
                return True
        for rm in _as_dict_list(blob.get("trail", {}).get("rooms")):
            for route in _as_dict_list(rm.get("routes")):
                p = route.get("puzzle") if isinstance(route.get("puzzle"), dict) else {}
                sol = (p.get("solution") or {}).get("answer")
                if sol and normalize_answer(sol) == ans_norm:
                    return True
    return False

# ───────────────────────── Archetypes (offline) ─────────────────────────

@dataclass
class Puzzle:
    id: str
    archetype: str
    prompt: str
    answer_format: Dict[str, Any]
    solution: Dict[str, Any]
    hints: List[str]
    decoys: List[str]
    paraphrases: List[str]
    # NEW: optional fields for mini-games and richer UIs
    mechanic: Optional[str] = None
    ui_spec: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "type": self.archetype,
            "archetype": self.archetype,
            "prompt": self.prompt,
            "answer_format": self.answer_format,
            "solution": self.solution,
            "hints": self.hints,
            "decoys": self.decoys,
            "paraphrases": self.paraphrases,
        }
        if self.mechanic:
            out["mechanic"] = self.mechanic
        if self.ui_spec:
            out["ui_spec"] = self.ui_spec
        return out

def _random_word(rng: random.Random, blacklist: set) -> str:
    pool = [w for w in ANAGRAM_WORDS if w not in blacklist]
    return rng.choice(pool) if pool else rng.choice(ANAGRAM_WORDS)

def gen_caesar(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    cands = [w for w in ANAGRAM_WORDS if 5 <= len(w) <= 8 and w not in blacklist]
    ans = rng.choice(cands) if cands else _random_word(rng, blacklist)
    k = rng.randrange(1,25); A = string.ascii_uppercase
    def enc(t,k): return "".join(A[(A.index(ch)+k)%26] if ch in A else ch for ch in t.upper())
    ct = enc(ans, k)
    return Puzzle(
        id=pid, archetype="caesar",
        prompt=f"A faded label reads: **{ct}**. Decode the Caesar shift.",
        answer_format={"pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": ans, "shift": k},
        hints=["The shift is not 13.","Try small positive shifts first."],
        decoys=[enc(ans,(k+1)%26), enc(ans,(k+2)%26)],
        paraphrases=[f"The inscription {ct} looks shifted.",
                     f"A rotating cipher hides a word: {ct}.",
                     f"Undo the shift on {ct}."]
    )

def gen_vigenere(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    key_pool = [w for w in ANAGRAM_WORDS if 5 <= len(w) <= 8 and w not in blacklist]
    key = rng.choice(key_pool) if key_pool else "VELVET"
    code_cands = [w for w in ANAGRAM_WORDS if 4 <= len(w) <= 8 and w not in blacklist and w.lower()!=key.lower()]
    code = rng.choice(code_cands) if code_cands else "EMBER"
    templates = [f"THE CODEWORD IS {code.upper()} HIDDEN BETWEEN LINES",
                 f"SEEK THE TOKEN {code.upper()} WITHIN THE NOTE",
                 f"{code.upper()} IS THE CLUE IN PLAIN SIGHT"]
    pt = rng.choice(templates); A = string.ascii_uppercase
    def enc(pt,k):
        out=[]; k=re.sub(r"[^A-Za-z]","",k).upper(); ki=0
        for ch in pt.upper():
            if ch in A:
                s=A.index(k[ki%len(k)]); out.append(A[(A.index(ch)+s)%26]); ki+=1
            else: out.append(ch)
        return "".join(out)
    ct = enc(pt,key)
    return Puzzle(
        id=pid, archetype="vigenere",
        prompt=f"Ciphered strip: **{ct}** (Vigenère). Submit the hidden CODEWORD.",
        answer_format={"pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": code, "key": key, "ciphertext": ct},
        hints=[f"The key is a material/gem ({len(key)} letters).","Focus on uppercase words."],
        decoys=[key.upper(), re.sub(r"[^A-Za-z]","",pt.replace(code.upper(),"TOKEN"))[:len(code)]],
        paraphrases=["Decrypt the strip to recover the embedded word.",
                     "The plaintext hides one word—submit that word.",
                     "A Vigenère line conceals the token."]
    )

def gen_numeric_lock(rng: random.Random, pid: str, blacklist: Optional[set] = None) -> Puzzle:
    # Normalize blacklist to lowercase/normalized tokens (works for digits too)
    bl = { (s if isinstance(s, str) else str(s)) for s in (blacklist or set()) }
    bl_norm = { normalize_answer(s) for s in bl }

    # Try a few times to avoid recently used codes
    for _ in range(25):
        d1, d2, d3, d4 = (rng.randrange(0, 10) for _ in range(4))
        code = f"{d1}{d2}{d3}{d4}"
        if normalize_answer(code) not in bl_norm:
            break

    c1 = f"The first two digits sum to {d1 + d2}."
    delta2 = d3 - d1
    c2 = f"The third digit is the first digit {'plus ' + str(delta2) if delta2 >= 0 else 'minus ' + str(abs(delta2))}."
    delta3 = d4 - d2
    c3 = f"The last digit equals the second digit {'plus ' + str(delta3) if delta3 >= 0 else 'minus ' + str(abs(delta3))}."

    return Puzzle(
        id=pid, archetype="numeric_lock",
        prompt=("A keypad blinks awaiting a 4-digit code.\n"
                f"- {c1}\n- {c2}\n- {c3}\nEnter the full code."),
        answer_format={"pattern": r"^\d{4}$"},
        solution={"answer": code},
        hints=["Write the constraints as equations.", "Solve first digits, then propagate."],
        decoys=[f"{(d1+1)%10}{d2}{d3}{d4}", f"{(d1-1)%10}{d2}{d3}{d4}"],
        paraphrases=["Compute the exact 4 digits.", "Three arithmetic clues define the code.",
                     "Derive the lock sequence."]
    )

# ───────────── Scene-aware mini games (safe & verifiable) ─────────────

def _scene_word_from_title(rng: random.Random, title: str, min_len=4, max_len=8) -> str:
    # try pulling a clean word from the title; otherwise fall back to our list
    toks = re.findall(r"[A-Za-z]{%d,%d}" % (min_len, max_len), (title or ""))
    toks = [t.lower() for t in toks if t.lower() not in {"the","and","room","hall","night","vault"}]
    return (rng.choice(toks) if toks else _random_word(rng, set())).lower()

def gen_acrostic(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    """First letters of each line spell the answer (fits the room theme)."""
    bl = {str(s).lower() for s in (blacklist or set())}

    # try a themed word first
    ans = re.sub(r"[^A-Za-z]", "", _scene_word_from_title(rng, theme)).lower()

    # reject if bad length or on the blacklist
    if not (4 <= len(ans) <= 8) or ans in bl:
        ans = _random_word(rng, bl)

    lines = []
    for ch in ans:
        frag = rng.choice([
            "shadows", "signal", "lantern", "archive", "glass", "stairs",
            "whisper", "copper", "fog", "vault", "console", "runes"
        ])
        lines.append(f"{ch.upper()}{ch.lower()}—{frag} tied to {theme or 'the room'}...")
    # 50% of the time, invert reading order (players start from bottom)
    bottom_start = rng.random() < 0.5
    poem = "\n".join(lines if not bottom_start else list(reversed(lines)))

    return Puzzle(
        id=pid, archetype="acrostic",
        prompt=(f"A scrap of verse is pinned to the wall:\n{poem}\n"
                + ("The scribbler inverted the stanza—start from the bottom line.\n" if bottom_start else "")
                + "What single word do the first letters spell?"),
        answer_format={"pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": ans},
        hints=["Read the FIRST letters vertically.",
               ("Start from the bottom line." if bottom_start else "Top line first, then down.")],
        decoys=[ans[::-1], "".join(sorted(ans)), ans[:-1]+ans[-1]*2],
        paraphrases=[f"Acrostic poem hints a word about {theme or 'this place'}."]
    )

# 5x5 Polybius/Tap code (I/J share a cell). We emit row-col pairs (1..5).
_POLY = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # J merged into I

# --- Variety: minimize repeating the same mechanic on the same route id
def _pz_mechanic_key(pz: Dict[str, Any]) -> str:
    """Normalize a puzzle's mechanic/type for comparison."""
    if not isinstance(pz, dict):
        return ""
    # Prefer explicit mechanic; fall back to legacy type/archetype
    m = (pz.get("mechanic") or pz.get("type") or pz.get("archetype") or "").lower()
    return m

def _reshuffle_mechanics_for_variety(room_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each scene, permute the three routes so that across scenes the same route id
    (cautious/brisk/risky) sees different mechanics when possible.
    This only affects the API payload; DB remains unchanged.
    """
    out = json.loads(json.dumps(room_json))  # deep copy of already-stripped payload
    trail = out.get("trail") or {}
    rooms = trail.get("rooms") or []

    # Track which mechanics each route id has already shown in prior scenes
    seen_by_route: Dict[str, set] = {"cautious": set(), "brisk": set(), "risky": set()}

    import itertools
    for rm in rooms:
        routes = list(rm.get("routes") or [])
        n = len(routes)
        if n < 2:
            # nothing to do
            for rt in routes:
                seen_by_route.setdefault(rt.get("id"), set()).add(_pz_mechanic_key((rt.get("puzzle") or {})))
            continue

        # Score all permutations by how many repeats they cause vs. seen_by_route
        perms = itertools.permutations(range(n))
        best_perm = None
        best_score = 10**9

        def score_perm(perm):
            score = 0
            for dest_idx, src_idx in enumerate(perm):
                dest_route_id = routes[dest_idx].get("id")
                mech = _pz_mechanic_key((routes[src_idx].get("puzzle") or {}))
                if mech in seen_by_route.get(dest_route_id, set()):
                    score += 1  # penalize repeating the same mechanic on the same route id
            return score

        for perm in perms:
            sc = score_perm(perm)
            if sc < best_score:
                best_score = sc
                best_perm = perm

        # Apply the best permutation (keeps the same 3 puzzles, just reassigns to the 3 route ids)
        # Apply best permutation by KEEPING each route id/label but SWAPPING the puzzles.
        if best_perm is not None:
            new_routes = []
            for dest_idx, src_idx in enumerate(best_perm):
                base  = json.loads(json.dumps(routes[dest_idx]))  # id/label/sub preserved
                donor = routes[src_idx]
                base["puzzle"] = donor.get("puzzle")              # <- puzzle reassigned
                new_routes.append(base)
            rm["routes"] = new_routes

        # Update seen set with what this scene now shows for each route id
        for rt in (rm.get("routes") or []):
            seen_by_route.setdefault(rt.get("id"), set()).add(_pz_mechanic_key((rt.get("puzzle") or {})))

    return out

def _tap_pairs_for_word(
    w: str,
    row_first: bool = True,
    row_labels: Optional[List[int]] = None,
    col_labels: Optional[List[int]] = None
) -> List[str]:
    """Encode a word into Polybius pairs with optional orientation/label tweaks."""
    out = []
    row_labels = row_labels or [1,2,3,4,5]
    col_labels = col_labels or [1,2,3,4,5]
    for ch in w.upper().replace("J", "I"):
        i = _POLY.index(ch)
        r = i // 5 + 1  # 1..5
        c = i % 5 + 1   # 1..5
        rr = row_labels[r-1]
        cc = col_labels[c-1]
        if row_first:
            out.append(f"{rr}-{cc}")
        else:
            out.append(f"{cc}-{rr}")
    return out

def gen_tapcode(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    bl = {str(s).lower() for s in (blacklist or set())}
    w = re.sub(r"[^A-Za-z]", "", _scene_word_from_title(rng, theme)).lower().replace("j", "i")
    if len(w) < 4 or w in bl:
        w = _random_word(rng, bl).lower().replace("j", "i")
    # Mild twists: sometimes column–row, and/or reversed counting
    row_first = rng.random() < 0.7          # 30% use column–row
    rev_rows  = rng.random() < 0.4
    rev_cols  = rng.random() < 0.4
    row_labels = [5,4,3,2,1] if rev_rows else [1,2,3,4,5]
    col_labels = [5,4,3,2,1] if rev_cols else [1,2,3,4,5]
    taps = ", ".join(_tap_pairs_for_word(w.upper(), row_first, row_labels, col_labels))
    mode_line = ("Pairs are ROW–COLUMN. " if row_first else "Pairs are COLUMN–ROW. ")
    row_line  = ("Rows count bottom→top (5→1). " if rev_rows else "Rows count top→bottom (1→5). ")
    col_line  = ("Cols count right→left (5→1). " if rev_cols else "Cols count left→right (1→5). ")
    return Puzzle(
        id=pid, archetype="tapcode",
        prompt=(f"From the {theme or 'pipes'}, you hear rhythmic taps: {taps}.\n"
                f"Decode with a 5×5 Polybius (I/J share a cell). {mode_line}{row_line}{col_line}"
                "Enter the word."),
        answer_format={"pattern": r"^[A-Za-z]{3,12}$"},
        solution={"answer": w.upper()},
        hints=["Map 1–5 row/col to letters in a 5×5 grid.", "Remember I/J share the same cell."],
        decoys=[w.upper()[::-1], (w[1:]+w[:1]).upper()],
        paraphrases=[f"Taps from the {theme or 'room'} encode a word."]
    )

def gen_pathcode(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    """Tiny grid; follow directions to read the path word."""
    bl = {str(s).lower() for s in (blacklist or set())}
    word = re.sub(r"[^A-Za-z]", "", _scene_word_from_title(rng, theme, 4, 6)).lower()
    if not (4 <= len(word) <= 6) or word in bl:
        word = _random_word(rng, bl)

    n = 3 if len(word) <= 5 else 4
    grid = [[rng.choice(string.ascii_uppercase) for _ in range(n)] for _ in range(n)]
    # Start can be anywhere for variety
    r, c = rng.randrange(0, n), rng.randrange(0, n)
    sr, sc = r, c                      # ← define sr, sc *after* picking r, c
    grid[r][c] = word[0].upper()
    # Allow L/U occasionally; optional wrap
    allow_LU = rng.random() < 0.6
    wrap = rng.random() < 0.35
    dirs = ["R","D"] + (["L","U"] if allow_LU else [])
    path: List[str] = []
    for ch in word[1:]:
        moves = []
        for d in rng.sample(dirs, k=len(dirs)):
            rr, cc = r, c
            if d == "R": cc = cc+1 if cc+1 < n else (0 if wrap else cc)
            if d == "L": cc = cc-1 if cc-1 >= 0 else (n-1 if wrap else cc)
            if d == "D": rr = rr+1 if rr+1 < n else (0 if wrap else rr)
            if d == "U": rr = rr-1 if rr-1 >= 0 else (n-1 if wrap else rr)
            if (rr, cc) != (r, c):
                moves.append((d, rr, cc))
        if not moves:
            # fallback: force right or down within bounds
            if c+1 < n: d, r, c = "R", r, c+1
            elif r+1 < n: d, r, c = "D", r+1, c
            else: d, r, c = "R", r, (c+1) % n
        else:
            d, r, c = rng.choice(moves)
        path.append(d)
        grid[r][c] = ch.upper()
    grid_str = "\n".join(" ".join(row) for row in grid)

    # Expose the start to the UI so we can visually mark it.
    ui_spec = {"grid": grid, "start": [sr, sc], "notes": "Follow the listed directions from the start."}

    return Puzzle(
        id=pid, archetype="pathcode",
        prompt=(f"A glowing tile grid is etched on the floor:\n{grid_str}\n"
                f"Start at the glowing letter (first in the word), then follow the path {', '.join(path)} to collect letters.\n"
                f"Directions are {', '.join(sorted(set(path)))}"
                f"{' (wrapping allowed)' if wrap else ''}. What word do you read?"),
        answer_format={"pattern": r"^[A-Za-z]{3,12}$"},
        solution={"answer": word.upper(), "grid": grid, "path": path, "start": [sr, sc]},
        hints=["Trace each step; record the letter you land on.",
               f"Grid size: {n}×{n}. Directions shown in the prompt."],
        decoys=[word.upper()[::-1], "".join(sorted(word.upper()))],
        paraphrases=[f"A path puzzle carved into {theme or 'the chamber'}."],
        mechanic="grid_input",
        ui_spec=ui_spec
    )

def gen_knightword(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    bl = {str(s).lower() for s in (blacklist or set())}
    cand = [w for w in ANAGRAM_WORDS if 5 <= len(w) <= 8 and w not in bl]
    word = rng.choice(cand) if cand else _random_word(rng, bl)
    n = 6 if len(word) >= 7 else 5

    K = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]

    start = (rng.randrange(0,n), rng.randrange(0,n))
    path = [start]
    for _ in word[1:]:
        r, c = path[-1]
        moves = [(r+dr, c+dc) for (dr,dc) in K
                 if 0 <= r+dr < n and 0 <= c+dc < n and (r+dr, c+dc) not in path]
        if not moves:
            start = (rng.randrange(0,n), rng.randrange(0,n))
            path = [start]
            continue
        path.append(rng.choice(moves))

    while len(path) < len(word):
        r, c = path[-1]
        moves = [(r+dr, c+dc) for (dr,dc) in K if 0 <= r+dr < n and 0 <= c+dc < n]
        path.append(rng.choice(moves))

    grid = [[rng.choice(string.ascii_uppercase) for _ in range(n)] for _ in range(n)]
    for (ch, (r,c)) in zip(word.upper(), path):
        grid[r][c] = ch
    grid_str = "\n".join(" ".join(row) for row in grid)

    row_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sr, sc = path[0]
    start_label = f"{row_labels[sr]}{sc+1}"
    ans = word.upper()
    decoys = [ans[::-1], "".join(sorted(ans)), (ans[1:]+ans[:1])]
    paraphrases = [
        "L-shaped leaps collect letters; begin from the marked square.",
        "Move like a chess knight to pick out a hidden word.",
        "Hop 2 then 1 at right angles; record the letters you land on."
    ]

    return Puzzle(
        id=pid,
        archetype="mini",
        prompt=(f"A tiled board in the {theme or 'room'} shows letters:\n{grid_str}\n"
                f"Start at {start_label}, moving like a knight (two then one). "
                "Collect the letters you land on and submit the word."),
        answer_format={"pattern": r"^[A-Za-z]{5,12}$"},
        solution={"answer": ans, "grid": grid, "start": start_label, "rule": "knight", "size": n},
        hints=["Move pattern: 2 then 1 at right angles.",
               f"Exactly {len(word)} letters; begin at {start_label}."],
        decoys=decoys,
        paraphrases=paraphrases,
        mechanic="grid_input",
        ui_spec={"grid": grid, "start": start_label, "notes": "Knight moves (2,1)."}
    )

def gen_valve_order(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    """
    Valve Precedence: Deduce the unique order of valves A..F/G from "X before Y" clues.
    Submit as A,B,C,....
    """
    import itertools

    n = rng.randint(6, 7)
    labels = list(string.ascii_uppercase[:n])  # ["A"..]
    # Hidden true order; we’ll emit constraints that force this exact order.
    true_order = labels[:]
    rng.shuffle(true_order)

    # edges are pairs (u, v) meaning u before v
    def is_valid(order, edges):
        pos = {ch: i for i, ch in enumerate(order)}
        return all(pos[u] < pos[v] for (u, v) in edges)

    def count_orders(edges):
        c = 0
        for perm in itertools.permutations(labels):
            if is_valid(perm, edges):
                c += 1
                if c > 1:  # early-out if not unique
                    break
        return c

    # seed with a few adjacent edges from the true order
    edges = []
    for i in range(n - 1):
        if rng.random() < 0.45:
            edges.append((true_order[i], true_order[i + 1]))
    if len(edges) < 2:
        edges = [(true_order[i], true_order[i + 1]) for i in range(0, n - 1, 2)][:2]

    # add cross-edges until there is exactly one consistent ordering
    tries = 0
    while count_orders(edges) != 1 and tries < 200:
        tries += 1
        i, j = sorted(rng.sample(range(n), 2))
        u, v = true_order[i], true_order[j]
        if (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))

    # rare fallback: use the full chain to guarantee uniqueness
    if count_orders(edges) != 1:
        edges = [(true_order[i], true_order[i + 1]) for i in range(n - 1)]

    bullets = "\n".join([f"- {u} before {v}" for (u, v) in edges])
    token_line = "Labels: " + ", ".join(labels)

    answer = ",".join(true_order)
    k = n

    # decoys
    rev  = list(reversed(true_order))
    rot  = true_order[1:] + true_order[:1]
    swap = true_order[:]
    if n >= 2:
        swap_idx = rng.randrange(0, n - 1)
        swap[swap_idx], swap[swap_idx + 1] = swap[swap_idx + 1], swap[swap_idx]

    return Puzzle(
        id=pid,
        archetype="mini",
        prompt=(f"A valve board in the {theme or 'room'} lists constraints. "
                "Determine the ONLY order that fits them all.\n"
                f"{bullets}\n{token_line}\n"
                "Use each label exactly once, from first to last."),
        answer_format={"pattern": rf"^(?:[A-{labels[-1]}],){{{k-1},{k-1}}}[A-{labels[-1]}]$"},
        solution={"answer": answer, "labels": labels, "edges": edges},
        hints=[f"Start by finding what must be earliest/latest.", f"Exactly {k} labels; each appears once."],
        decoys=[",".join(rev), ",".join(rot), ",".join(swap)],
        paraphrases=["Order the valves so all 'before' relations hold."],
        mechanic="sequence_input",
        ui_spec={"sequence": labels, "notes": "Tap labels in order or type A,B,C,…"}
    )

def gen_translate_with_legend(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    tokens = SEQ_TOKENS[:]
    k = rng.randrange(7, 11)
    cues = [("short beep","tap"), ("long beep","hold"),
            ("hiss","left"), ("clank","right"),
            ("gust","up"), ("drip","down"),
            ("rumble","rotate_left"), ("chime","rotate_right")]
    rng.shuffle(cues)
    legend = cues[:5]
    legend_map = {name: action for name, action in legend}

    seq_actions = [rng.choice(list(legend_map.values())) for _ in range(k)]
    seq_cues    = [rng.choice([n for n,a in legend if a == act]) for act in seq_actions]

    ans = ",".join(seq_actions)
    rev = ",".join(reversed(seq_actions))
    rot = ",".join(seq_actions[1:] + seq_actions[:1])
    tweaked = seq_actions[:]
    tweak_idx = rng.randrange(0, k)
    alt = rng.choice([t for t in legend_map.values() if t != seq_actions[tweak_idx]])
    tweaked[tweak_idx] = alt

    legend_lines = "\n".join([f"- {n} → {a}" for n,a in legend])

    return Puzzle(
        id=pid,
        archetype="mini",
        prompt=(f"From the {theme or 'consoles'}, signals repeat:\n"
                f"{', '.join(seq_cues)}\n"
                "Use this legend to translate each cue into an action:\n"
                f"{legend_lines}\n"
                "Enter the exact action sequence using the chips."),
        answer_format={"pattern": r"^[A-Za-z0-9,\-]{5,200}$"},
        solution={"answer": ans, "legend": legend, "length": k},
        hints=[f"Sequence length: {k}.", f"Example: “{legend[0][0]}” = {legend[0][1]}. Order matters."],
        decoys=[rev, rot, ",".join(tweaked)],
        paraphrases=[
            "The panel repeats audio cues; translate each into its action.",
            "Use the legend to map every sound to a control input."
        ],
        mechanic="sequence_input",
        ui_spec={"sequence": SEQ_TOKENS[:]})

def gen_scene_mini(rng, pid, blacklist, theme=""):
    choices = [
        (gen_knightword,           3),
        (gen_valve_order,          3),
        (gen_signal_translate,     1),  # cue-only audio memory
        (gen_translate_with_legend,1),
        (gen_pathcode,             2),
        (gen_tapcode,              2),
        (gen_acrostic,             1),
    ]
    funcs, w = zip(*choices)
    f = rng.choices(funcs, weights=w, k=1)[0]
    return f(rng, pid, blacklist, theme)

# --- Sanitizer helpers -------------------------------------------------

def _regen_puzzle(archetype: str, rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Dict[str, Any]:
    if archetype == "acrostic":  return gen_acrostic(rng, pid, blacklist, theme).to_json()
    if archetype == "tapcode":   return gen_tapcode(rng, pid, blacklist, theme).to_json()
    if archetype == "pathcode":  return gen_pathcode(rng, pid, blacklist, theme).to_json()
    return gen_scene_mini(rng, pid, blacklist, theme).to_json()

_ANAGRAM_TOKEN_RE = re.compile(r"(?:'|\*\*)([A-Za-z]{3,12})(?:'|\*\*)")

# NEW: ensure sequence minis explain what the chips mean in-world
def _inject_sequence_legend(p: Dict[str, Any], theme: str = "") -> None:
    if (p.get("type") != "mini") or (p.get("mechanic") != "sequence_input"):
        return
    prompt = p.get("prompt") or ""
    # if author already provided a legend/controls, keep it
    if re.search(r"\blegend\b|\bcontrols?:", prompt, re.I):
        return
    tokens = (p.get("ui_spec") or {}).get("sequence") or []
    if tokens and not all(t in SEQ_TOKENS for t in tokens):
        return
    legend_lines = (
        "Controls:\n"
        "- tap = press the brass button\n"
        "- hold = keep the button pressed\n"
        "- left/right = nudge the lever left/right\n"
        "- up/down = raise/lower the slider\n"
        "- rotate_left/right = turn the crank counter/clockwise"
    )
    p["prompt"] = (prompt.rstrip() + "\n" + legend_lines).strip()

# NEW: For grid_input "collect letters" puzzles, force a single-letter grid
# that actually contains the word answer at least once.
def _force_letter_grid_for_answer(rng: random.Random, p: Dict[str, Any], answer: str) -> None:
    ans = re.sub(r"[^A-Za-z]", "", str(answer or "")).upper()
    if not (4 <= len(ans) <= 12):
        return
    n = 3 if len(ans) <= 5 else 4
    grid = [[rng.choice(string.ascii_uppercase) for _ in range(n)] for _ in range(n)]

    # Place each letter of the answer somewhere in the grid.
    coords = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(coords)
    for ch in ans[: min(len(coords), len(ans))]:
        r, c = coords.pop()
        grid[r][c] = ch

    ui = p.setdefault("ui_spec", {})
    ui["grid"] = grid

    # Make the prompt explicit and add clear, non-spoiler hints.
    hints = [h for h in (p.get("hints") or []) if isinstance(h, str) and h.strip()]
    extra = [f"Grid size: {n}×{n}.", f"Collect exactly {len(ans)} letters."]
    p["hints"] = (hints + [h for h in extra if h not in hints])[:2]

# ── MC visual/interactive helpers ────────────────────────────────────────────
_ROMAN_MAP = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
def _roman_to_int(s: str) -> Optional[int]:
    s = re.sub(r"[^IVXLCDM]", "", (s or "").upper())
    if not s: return None
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN_MAP.get(ch, 0)
        if v < prev: total -= v
        else: total += v; prev = v
    return total if total > 0 else None

def _reroll_recent_mini(rng: random.Random, p: Dict[str, Any], blacklist: set, theme: str = "") -> Dict[str, Any]:
    """
    If a mini-game's solution was used within the cooldown window, replace it.
    For MC, try switching the correct option; otherwise regenerate same-flavor.
    Returns a (possibly) new puzzle dict.
    """
    ans = str((p.get("solution") or {}).get("answer") or "")
    if not answer_recently_used(ans):
        return p

    mech = (p.get("mechanic") or "").strip().lower()
    ui   = p.setdefault("ui_spec", {})

    # MC: switch the correct option to a non-recent one if available
    if mech == "multiple_choice":
        opts = _normalize_mc_options(ui.get("options") or [])
        candidates = [o for o in opts
                      if not answer_recently_used(o["label"]) and not answer_recently_used(o["id"])]
        if candidates:
            pick = rng.choice(candidates)
            p.setdefault("solution", {})["answer"] = pick["label"]
            return p
        # fall through to regeneration

    # Regenerate another mini of the same broad flavor when possible
    pid   = p.get("id") or "mini_autogen"
    kind  = _classify_mini_kind(p)
    if   kind == "signal":      q = gen_signal_translate(rng, pid, blacklist, theme)
    elif kind == "legend":      q = gen_translate_with_legend(rng, pid, blacklist, theme)
    elif kind == "pathcode":    q = gen_pathcode(rng, pid, blacklist, theme)
    elif kind == "knightword":  q = gen_knightword(rng, pid, blacklist, theme)
    elif kind == "tapcode":     q = gen_tapcode(rng, pid, blacklist, theme)
    else:                       q = gen_acrostic(rng, pid, blacklist, theme)
    return q.to_json()

def _normalize_mc_options(opts: List[Any]) -> List[Dict[str, Any]]:
    """Allow strings or dicts; normalize to {id,label, sprite?, image?, meta?}."""
    out = []
    for o in (opts or []):
        if isinstance(o, dict):
            label = str(o.get("label") or o.get("id") or "").strip()
            oid   = str(o.get("id") or normalize_answer(label)).strip()
            out.append({"id": oid, "label": label, **{k:v for k,v in o.items() if k in {"sprite","image","meta"}}})
        else:
            label = str(o).strip()
            out.append({"id": normalize_answer(label), "label": label})
    return out

def _embellish_visual_mc(rng: random.Random, p: Dict[str, Any], theme: str = "") -> None:
    """
    If a multiple_choice prompt references silhouettes/shadows or tools,
    attach sprite info and (when a Roman numeral is present) a shadow-length
    rule the client can render/measure. Does NOT change verification.
    """
    if (p.get("type") != "mini") or (p.get("mechanic") != "multiple_choice"):
        return
    ui = p.setdefault("ui_spec", {})
    opts_raw = ui.get("options") or []
    if not isinstance(opts_raw, list) or len(opts_raw) < 4:
        return

    prompt = (p.get("prompt") or "")
    text_l = prompt.lower()

    # 1) Add silhouettes if none are provided yet
    needs_sprites = not any(isinstance(o, dict) and (o.get("sprite") or o.get("image")) for o in opts_raw)
    if needs_sprites and re.search(r"shadow|silhouette|tool|cabinet|hammer|drawer", text_l):
        pack = "workshop_tools_v1"  # front-end sprite pack (svg/png set)
        opts = _normalize_mc_options(opts_raw)
        for o in opts:
            o["sprite"] = f"{pack}/{o['label'].lower().replace(' ', '_')}.svg"
        ui["options"] = opts
        ui["sprite_pack"] = pack
        ui.setdefault("notes", "Tap a tool to see its silhouette and shadow.")

    # 2) If a Roman numeral is present (e.g., VII), add a measurable shadow rule
    m = re.search(r"\b([IVXLCDM]+)\b", prompt)
    val = _roman_to_int(m.group(1)) if m else None
    if val:
        # Determine which option is correct
        ans = str((p.get("solution") or {}).get("answer") or "")
        opts = _normalize_mc_options(ui.get("options") or opts_raw)
        # Build lengths: correct option = target; other options = near misses
        near = [max(1, val-2), max(1, val-1), val+1, val+2]
        rng.shuffle(near)
        lengths = {}
        for o in opts:
            is_ans = normalize_answer(o["label"]) == normalize_answer(ans) or normalize_answer(o["id"]) == normalize_answer(ans)
            lengths[o["id"]] = val if is_ans else near.pop() if near else max(1, val + rng.choice([-2,-1,1,2]))
        ui["shadow_rule"] = {
            "units": "marks",
            "target": val,
            "angle_deg": 45,             # client can animate if desired
            "lengths": lengths           # id -> integer (for on-hover / measure)
        }
        # tighten hints for the interactive affordance
        hints = [h for h in (p.get("hints") or []) if h]
        add = ["Use the ruler to measure each shadow.", f"Target length: {val} marks."]
        p["hints"] = (hints + [h for h in add if h not in hints])[:2]

def _sanitize_trail_puzzles(room: Dict[str, Any], rng: random.Random, blacklist: set) -> None:
    trail = room.get("trail") or {}
    rooms = trail.get("rooms") or []
    for rm in rooms:
        theme = rm.get("title", "") or rm.get("text", "")
        routes = rm.get("routes") or []
        for rt in routes:
            p = rt.get("puzzle")
            pid = f"{rm.get('id','room')}_{rt.get('id','route')}_pz"

            if not isinstance(p, dict):
                rt["puzzle"] = _synth_puzzle(rng, pid, blacklist, theme)
                continue

            # normalize id/type
            p["id"] = p.get("id") or pid
            typ = (p.get("type") or p.get("archetype") or "").lower()

            if typ == "mini":
                # ensure mechanic + ui_spec exist
                mech = (p.get("mechanic") or "").lower()
                if mech not in ALLOWED_MECHANICS:
                    p["mechanic"] = "text_input"
                    mech = "text_input"
                ui = p.setdefault("ui_spec", {})

                if mech == "multiple_choice" and _looks_trivial_multiple_choice(p):
                    rt["puzzle"] = gen_pathcode(rng, p.get("id") or pid, blacklist, theme).to_json()
                    continue

                # NEW: enrich visual MCs (silhouettes + shadow measuring when applicable)
                if mech == "multiple_choice":
                    try:
                        _embellish_visual_mc(rng, p, theme)
                    except Exception as e:
                        try: current_app.logger.warning("[escape] MC embellish skipped: %s", e)
                        except Exception: pass

                sol = (p.get("solution") or {}).get("answer", "")

                # --- Sequence minis: guarantee tokens, length ≥5, and visible cues/series
                if mech == "sequence_input":
                    cue_only = bool(ui.get("cue_set") or ui.get("cues_audio") or ui.get("cues"))
                    if not ui.get("sequence") and not cue_only:
                        ui["sequence"] = SEQ_TOKENS[:]

                    ans = (p.get("solution") or {}).get("answer", "")
                    if isinstance(ans, list):
                        ans = ",".join(str(t).strip() for t in ans if str(t).strip())
                        p.setdefault("solution", {})["answer"] = ans
                    steps = [t for t in re.split(r"[,\s]+", str(ans)) if t]

                    def _has_explicit_series(txt: str, tokens: List[str]) -> bool:
                        t = re.sub(r"(?is)controls:\s*[-•].*$", "", txt or "")
                        token_pat = r"(?:%s)" % "|".join(re.escape(x) for x in tokens)
                        return bool(re.search(r"\b\d\s*-\s*\d\b", t)) or bool(
                            re.search(rf"\b{token_pat}\b(?:\s*,\s*\b{token_pat}\b){{2,}}", t)
                        )

                    has_series = True if cue_only else _has_explicit_series(p.get("prompt",""), ui.get("sequence", []))

                    if (len(steps) < 5) or (not has_series):
                        rt["puzzle"] = gen_signal_translate(rng, pid, set(), theme).to_json()
                        p  = rt["puzzle"]
                        mech = "sequence_input"
                        ui = p.setdefault("ui_spec", {})
                        # recompute from the NEW puzzle
                        ans = (p.get("solution") or {}).get("answer", "")
                        steps = [t for t in re.split(r"[,\s]+", str(ans)) if t]
                        if len(steps) < 5:
                            try:
                                rt["puzzle"] = gen_signal_translate(rng, pid, set(), theme).to_json()
                            except Exception as e:
                                current_app.logger.warning("[escape] gen_signal_translate failed: %s; falling back", e)
                                rt["puzzle"] = gen_pathcode(rng, pid, set(), theme).to_json()
                            continue

                # --- Grid minis: fix invalid grids / multi-char cells for letter collection
                if mech == "grid_input":
                    grid = ui.get("grid")
                    prompt_txt = p.get("prompt") or ""
                    needs_letters = bool(re.search(r"\b(letter|letters|spell|word|collect)\b", prompt_txt, re.I))
                    bad_grid = not (isinstance(grid, list) and grid and all(isinstance(r, list) for r in grid))
                    long_tokens = any(len(str(cell)) > 1 for row in (grid or []) for cell in row) if not bad_grid else False
                    sol = (p.get("solution") or {}).get("answer", "")

                    if bad_grid or (needs_letters and long_tokens):
                        if re.fullmatch(r"^[A-Za-z]{4,12}$", re.sub(r"[^A-Za-z]", "", str(sol))):
                            _force_letter_grid_for_answer(rng, p, sol)
                            p["prompt"] = (prompt_txt.rstrip() + "\nTap tiles to collect letters that spell the word.").strip()
                        else:
                            rt["puzzle"] = gen_knightword(rng, p.get("id") or pid, blacklist, theme).to_json()
                            continue

                    # enforce ≥4 taps
                    if len(re.sub(r"[^A-Za-z0-9]", "", str(sol) or "")) < 4:
                        rt["puzzle"] = gen_knightword(rng, p.get("id") or pid, blacklist, theme).to_json()
                        continue

                # --- Coerce/repair pattern so it actually matches the current solution
                sol_now = (p.get("solution") or {}).get("answer", "")
                af = p.get("answer_format") or {}
                pat = (af.get("pattern") or "").strip()
                need = _default_pattern_for_answer(sol_now)
                if (not pat) or (not re.fullmatch(r"^\^.*\$$", pat)) or not (
                    re.fullmatch(pat, str(sol_now)) or re.fullmatch(pat, normalize_answer(str(sol_now)))
                ):
                    af["pattern"] = need
                p["answer_format"] = af

                # Add scene-grounding legend for sequences when no cue_set; tighten hints
                if not (p.get("mechanic") == "sequence_input" and (p.get("ui_spec") or {}).get("cue_set")):
                    _inject_sequence_legend(p, theme)
                _upgrade_minigame_hints(p)

                # NEW: avoid cooldown collisions before validation later
                try:
                    p = _reroll_recent_mini(rng, p, blacklist, theme)
                except Exception as e:
                    try: current_app.logger.warning("[escape] reroll_recent_mini skipped: %s", e)
                    except Exception: pass

                rt["puzzle"] = p
                continue

            # legacy allowed → rebuild deterministically to keep things valid
            if typ == "acrostic":
                rt["puzzle"] = gen_acrostic(rng, p["id"], blacklist, theme).to_json()
            elif typ == "tapcode":
                rt["puzzle"] = gen_tapcode(rng, p["id"], blacklist, theme).to_json()
            elif typ == "pathcode":
                rt["puzzle"] = gen_pathcode(rng, p["id"], blacklist, theme).to_json()
            else:
                rt["puzzle"] = _synth_puzzle(rng, p["id"], blacklist, theme)

def _annotate_grid_path_meta(p: Dict[str, Any]) -> None:
    if (p.get("mechanic") != "grid_input"): return
    ui = p.get("ui_spec") or {}
    grid = ui.get("grid")
    if not (isinstance(grid, list) and grid and isinstance(grid[0], list)): return

    # Try to parse answers like "G5G8G9G6G3G2" into labels present on the grid
    labels = [str(cell) for row in grid for cell in row]
    lab_re = re.compile("|".join(sorted(map(re.escape, labels), key=len, reverse=True)))
    ans = str((p.get("solution") or {}).get("answer", ""))
    path = lab_re.findall(ans)
    if len(path) < 4: return

    start, end, taps = path[0], path[-1], len(path)
    clar = f"\nStart at {start}; end at {end}. Move only to adjacent tiles (no diagonals). Exactly {taps} taps."
    if clar.strip() not in (p.get("prompt") or ""):
        p["prompt"] = (p.get("prompt","").rstrip() + clar).strip()
    # Tighten hints so players aren’t guessing
    hints = [h for h in (p.get("hints") or []) if h]
    needed = ["Tap sequential adjacent tiles; don’t skip.", f"Path length: {taps} (from {start} to {end})."]
    p["hints"] = (hints + [h for h in needed if h not in hints])[:2]

# ───────────────────────── Fragment rules ─────────────────────────

def apply_fragment_rule(answer: str, rule: str) -> str:
    ans = normalize_answer(answer)
    rule = (rule or "").strip().upper()

    if rule.startswith("CONST:"):
        return normalize_answer(rule.split("CONST:",1)[1])[:4]

    if rule == "FIRST2":  return ans[:2]
    if rule == "FIRST3":  return ans[:3]
    if rule == "LAST2":   return ans[-2:]
    if rule == "LAST3":   return ans[-3:]

    m = re.match(r"CAESAR:\+?(-?\d+);(FIRST2|FIRST3|LAST2|LAST3)$", rule)
    if m:
        k = int(m.group(1)) % 26
        A = string.ascii_uppercase
        shifted = "".join(A[(A.index(ch)+k)%26] if ch in A else ch for ch in ans)
        sub = m.group(2)
        return apply_fragment_rule(shifted, sub)

    m = re.match(r"IDX:([0-9,]+)$", rule)
    if m:
        idxs = [int(x) for x in m.group(1).split(",") if x.isdigit()]
        out = "".join(ans[i] for i in idxs if 0 <= i < len(ans))
        return out[:4]

    if rule == "NUM:LAST2":
        digits = re.sub(r"\D","",answer)
        return (digits[-2:] if digits else "00")

    return ans[:2]

def is_valid_fragment_rule(rule: str) -> bool:
    return bool(re.match(
        r"^(FIRST2|FIRST3|LAST2|LAST3|CONST:[A-Za-z0-9]{1,4}|CAESAR:\+?-?\d+;(FIRST2|FIRST3|LAST2|LAST3)|IDX:[0-9,]{1,8}|NUM:LAST2)$",
        (rule or "").strip().upper()
    ))

# ───────────────────────── LLM wiring ─────────────────────────

def _get_openai_client():
    if os.getenv("ESCAPE_MODEL","").lower()=="off" or os.getenv("ESCAPE_FORCE_OFFLINE","").lower() in ("1","true","yes"):
        return None, None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return None, None
    try:
        from openai import OpenAI  # modern
        return OpenAI(api_key=api_key), "modern"
    except Exception:
        pass
    try:
        import openai  # legacy
        openai.api_key = api_key
        return openai, "legacy"
    except Exception:
        return None, None

def _default_pattern_for_type(p_type: str, answer: str) -> str:
    p = (p_type or "").lower()
    # our three archetypes are word answers
    if p in {"acrostic", "tapcode", "pathcode"}:
        return r"^[A-Za-z]{3,12}$"
    # fallback: match the answer shape if it's numeric
    if re.fullmatch(r"^\d+$", str(answer or "")):
        return r"^\d+$"
    return r"^[A-Za-z]{3,12}$"

def _synth_puzzle(rng: random.Random, pid: str, blacklist_lower: Optional[set] = None,
                  theme: str = "") -> Dict[str, Any]:
    bl = blacklist_lower or set()
    return gen_scene_mini(rng, pid, bl, theme).to_json()

def _coerce_same_fragment_or_const_all(rule: str, puzzles: List[Dict[str,Any]]) -> Tuple[str, str]:
    """Ensure *all* routes for a room yield the same fragment; fallback to CONST if not."""
    if not is_valid_fragment_rule(rule):
        rule = "FIRST2"
    frags = []
    for p in puzzles:
        ans = (p.get("solution") or {}).get("answer", "")
        frags.append(apply_fragment_rule(ans, rule))
    first = frags[0] if frags else ""
    if any(f != first for f in frags):
        rule = f"CONST:{first}"
        frags = [first] * len(frags)
    return rule, first

def _replace_recent_answers(blob: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    recent_norm = _recent_answer_set()
    recent_lower = {s.lower() for s in recent_norm}

    trail = blob.get("trail") or {}
    rooms = (trail.get("rooms") or [])
    for i, rm in enumerate(rooms, start=1):
        theme = rm.get("title","") or rm.get("text","")
        routes = (rm.get("routes") or [])
        for j, rt in enumerate(routes, start=1):
            p = (rt.get("puzzle") or {})
            if not isinstance(p, dict):
                rt["puzzle"] = _synth_puzzle(rng, f"r{i}_auto_{j}", recent_lower, theme)
                continue
            ans = (p.get("solution") or {}).get("answer", "")
            if normalize_answer(ans) in recent_norm:
                pid = p.get("id") or f"r{i}_auto_{j}"
                typ = (p.get("type") or p.get("archetype") or "").lower()
                if   typ == "acrostic": new_p = gen_acrostic(rng, pid, recent_lower, theme).to_json()
                elif typ == "tapcode":  new_p = gen_tapcode(rng,  pid, recent_lower, theme).to_json()
                elif typ == "pathcode": new_p = gen_pathcode(rng, pid, recent_lower, theme).to_json()
                else:                   new_p = _synth_puzzle(rng, pid, recent_lower, theme)
                new_p["id"] = pid
                rt["puzzle"] = new_p

        # re-coerce fragments across all routes
        # re-coerce fragments across all routes (tolerant; sanitize happens later)
        puzzles = []
        for r in routes:
            q = r.get("puzzle") if isinstance(r.get("puzzle"), dict) else {}
            if q:
                # Do NOT validate here; LLM output may be unsanitized yet.
                puzzles.append(q)
        if puzzles:
            fr_rule = rm.get("fragment_rule") or "FIRST2"
            fr_rule, _ = _coerce_same_fragment_or_const_all(fr_rule, puzzles)
            rm["fragment_rule"] = fr_rule


    blob["trail"] = {**trail, "rooms": rooms}
    return blob

# AFTER: new helper
def _thematic_routes_for_room(rng: random.Random, theme: str) -> Dict[str, Tuple[str, str]]:
    """
    Returns per-route (label, sub) that reference the scene.
    Label = button text; sub = short pros/cons/hinty line.
    """
    noun = re.sub(r"[^A-Za-z ]", "", (_scene_word_from_title(rng, theme, 4, 12) or "")).strip().lower() or "device"
    art = "an" if noun[:1] in "aeiou" else "a"

    cautious_verbs = ["Inspect", "Observe", "Trace", "Skirt"]
    brisk_verbs    = ["Weave past", "Cut through", "Stride by", "Skim past"]
    risky_verbs    = ["Tamper with", "Force", "Gamble on", "Rush"]

    c_lab = f"{rng.choice(cautious_verbs)} the {noun}"
    b_lab = f"{rng.choice(brisk_verbs)} the {noun}"
    r_lab = f"{rng.choice(risky_verbs)} the {noun}"

    return {
        "cautious": (c_lab, "Safer path — hints unlock sooner."),
        "brisk":    (b_lab,  "Faster path — hints unlock +30s; small time bonus if clean."),
        "risky":    (r_lab,  "Bold path — hints unlock +60s; bigger bonus if first try.")
    }

def _ensure_routes(rm: Dict[str, Any], r_index: int, rng: random.Random, count: int = 3) -> Dict[str, Any]:
    theme = f"{rm.get('title','')} {rm.get('text','')}".strip()
    routes_raw = _as_dict_list(rm.get("routes"))
    cleaned = []

    # Normalize any provided routes/puzzles first
    for rt in routes_raw:
        p = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
        if not p:
            p = _synth_puzzle(rng, f"r{r_index}_autopz_{len(cleaned)+1}", theme=theme)
        _upgrade_minigame_hints(p)
        if "id" not in p:
            p["id"] = f"r{r_index}_autopz_{len(cleaned)+1}"
        if "type" not in p and "archetype" in p:
            p["type"] = p["archetype"]
        if "answer_format" not in p:
            p["answer_format"] = {
                "pattern": _default_pattern_for_type(
                    p.get("type"), (p.get("solution") or {}).get("answer", "")
                )
            }
        if "solution" not in p or (p.get("solution") or {}).get("answer") in (None, ""):
            p = _synth_puzzle(rng, f"r{r_index}_autopz_{len(cleaned)+1}", theme=theme)
        _upgrade_minigame_hints(p)

        cleaned.append({
            "id": (rt.get("id") or "").lower(),
            "label": rt.get("label"),
            "sub": rt.get("sub"),
            "puzzle": p
        })

    # Build the final 3 routes (fill/rename/theme)
    ids = ["cautious", "brisk", "risky"][:count]
    themed = _thematic_routes_for_room(rng, theme)

    out = []
    for rid in ids:
        existing = next((x for x in cleaned if x["id"] == rid), None)
        label, sub = themed.get(rid, (None, None))
        if existing:
            lbl = existing.get("label") or label or rid.title()
            out.append({
                "id": rid,
                "label": lbl,
                "sub": existing.get("sub") or sub,
                "puzzle": existing["puzzle"]
            })
        else:
            pid = f"r{r_index}_{rid}_pz_autogen"
            out.append({
                "id": rid,
                "label": label or rid.title(),
                "sub": sub,
                "puzzle": _synth_puzzle(rng, pid, theme=theme)
            })

    rm["routes"] = out
    fr = rm.get("fragment_rule") or "FIRST2"
    rm["fragment_rule"] = fr if is_valid_fragment_rule(fr) else "FIRST2"
    return rm

def _synth_room(idx: int, rng: random.Random) -> Dict[str, Any]:
    theme = f"Room {idx}"
    p1 = _synth_puzzle(rng, f"r{idx}_caut_pz", theme=theme)
    p2 = _synth_puzzle(rng, f"r{idx}_brisk_pz", theme=theme)
    p3 = _synth_puzzle(rng, f"r{idx}_risky_pz", theme=theme)

    # AFTER
    fr, _ = _coerce_same_fragment_or_const_all("FIRST2", [p1, p2, p3])
    theme = f"Waystation {idx}"
    themed = _thematic_routes_for_room(rng, theme)
    return {
        "id": f"room_{idx}",
        "title": theme,
        "text": "",
        "routes": [
            {"id": "cautious", "label": themed["cautious"][0], "sub": themed["cautious"][1], "puzzle": p1},
            {"id": "brisk",    "label": themed["brisk"][0],    "sub": themed["brisk"][1],    "puzzle": p2},
            {"id": "risky",    "label": themed["risky"][0],    "sub": themed["risky"][1],    "puzzle": p3},
        ],
        "fragment_rule": fr,
    }

def _sanitize_llm_blob(blob: Dict[str, Any], rng: random.Random, date_key: str) -> Dict[str, Any]:
    """Force LLM output into exactly-3-rooms, exactly-3-routes-per-room."""
    if "trail" not in blob and "rooms" in blob:
        blob["trail"] = {"supplies_start": blob.get("supplies_start", SUPPLIES_START_DEFAULT),
                         "rooms": blob.get("rooms", [])}

    trail = blob.get("trail")
    if not isinstance(trail, dict): trail = {}
    rooms = _as_dict_list(trail.get("rooms"))

    rooms = rooms[:3]
    while len(rooms) < 3:
        rooms.append(_synth_room(len(rooms)+1, rng))

    fixed_rooms = []
    for i, rm in enumerate(rooms, start=1):
        if not isinstance(rm, dict): rm = {"id": f"room_{i}", "routes": []}
        rm.setdefault("id", f"room_{i}")
        rm.setdefault("title", f"Waystation {i}")
        rm.setdefault("text", "")
        fixed_rooms.append(_ensure_routes(rm, i, rng, 3))

    trail["rooms"] = fixed_rooms
    blob["trail"] = trail

    # Ensure final prompt + pattern exist (answer gets reconciled in validate_trailroom)
    final = blob.get("final")
    if not isinstance(final, dict): final = {}
    final.setdefault("id", "final")
    final.setdefault("prompt", "Assemble the three fragments to form the PASSCODE.")
    af = final.get("answer_format")
    if not isinstance(af, dict): af = {}
    af.setdefault("pattern", r"^[A-Za-z0-9]{4,12}$")
    final["answer_format"] = af
    blob["final"] = final

    # Basic top fields
    blob.setdefault("id", date_key)
    blob.setdefault("title", "Daily Trail")
    blob.setdefault("intro", "Three stops, one final lock.")
    blob.setdefault("difficulty", blob.get("difficulty", DEFAULT_DIFFICULTY))
    blob.setdefault("anti_spoiler", {"paraphrase_variants": 3, "decoys": 2})
    if not isinstance(blob.get("npc_lines"), list): blob["npc_lines"] = []

    return blob

def _trail_prompt(date_key: str) -> str:
    # recent context to discourage repeats
    recent_titles = []
    try:
        for r in recent_rooms(RECENT_WINDOW_DAYS):
            t = (r.json_blob or {}).get("title")
            if t: recent_titles.append(t)
    except Exception:
        pass
    recent_titles = recent_titles[:8]

    try:
        ban_answers = sorted(list(_recent_answer_set()))[:60]
    except Exception:
        ban_answers = []

    recent_titles_s = " • ".join(recent_titles) if recent_titles else "(none)"
    ban_answers_s  = ", ".join(ban_answers) if ban_answers else "(none)"

    return (
        "You are designing TODAY’S daily escape: **exactly 3 scenes → 1 final lock**.\n"
        "\n"
        "BIG IDEA: Each scene contains a **brand-new micro-game** that is easy to learn (<30s), fast to play (≈1–2 min),\n"
        "and thematically tied to the scene. These are not recycled puzzle types: invent fresh rules/skins daily,\n"
        "but express them using one of FOUR frontend mechanics so the client can render them:\n"
        "  mechanics = multiple_choice | sequence_input | grid_input | text_input\n"
        "\n"
        "HARD REQUIREMENTS:\n"
        "• JSON ONLY, follow the schema below precisely; no additional prose or commentary.\n"
        "• EXACTLY 3 scenes in order (rooms[0..2]).\n"
        "• Each scene has EXACTLY three routes with ids: cautious, brisk, risky.\n"
        "• For every route, produce a **mini-game**: {type:'mini', mechanic, ui_spec, prompt, answer_format, solution.answer, hints}.\n"
        "• `prompt` must reference concrete props of THAT scene so it feels bespoke.\n"
        "• Keep prompts tight (≈40–80 words).\n"
        "• Provide EXACTLY 2 hints per puzzle:\n"
        "  - Hint 1 = mechanic nudge (how to interact / what to notice).\n"
        "  - Hint 2 = concrete anchor (length, mapping example, grid size). No spoilers.\n"
        "• sequence_input: ui_spec = {\"sequence\": [str, ...]} (tokens like 'tap','hold','left','right','rotate_left').\n"
        "  Hints SHOULD include the exact step count and one mapping example when appropriate.\n"
        "• The three routes in the SAME scene must share **one fragment_rule** that extracts the same fragment from each answer.\n"
        "  Prefer FIRST2/LAST2; FIRST3/LAST3/IDX:i,j are allowed if thematic.\n"
        "• Final code = concatenation of the three scene fragments. Final answer must be a 4–12 char alphanumeric string.\n"
        "and thematically tied to the scene. Prefer **grid_input** and **sequence_input**.\n"
        "At most **one** multiple_choice across all three scenes, and only if it requires deduction from ≥2 clues.\n"
        "Minimum interaction: grid_input must require ≥4 taps; sequence_input length must be 5–9.\n"
        "Never ask for a raw fact (e.g., “Which word names that number?”).\n"
        "• DO NOT use or hint at any words in BAN_ANSWERS. Avoid stock riddle answers entirely (echo, time, shadow, piano, etc.).\n"
        "\n"
        f"RECENT_TITLES (avoid similarity): {recent_titles_s}\n"
        f"BAN_ANSWERS: {ban_answers_s}\n"
        "\n"
        "MECHANIC RULES (ui_spec):\n"
        "• multiple_choice: ui_spec = {\"options\": [str, ...]} and solution.answer must equal one of the options.\n"
        "• grid_input: ui_spec = {\"grid\": [[str,...],...]} 2..6 rows; each cell is a 1–2 char label. Optional: start, goal, notes.\n"
        "• text_input: ui_spec = {} (freeform short token answer).\n"
        "\n"
        "ANCHOR PATTERNS:\n"
        "• answer_format.pattern must be anchored with ^...$ and match the answer (use ^[A-Za-z]{2,16}$ for words or ^\\d{1,12}$ for numbers;\n"
        "  for sequences allow ^[A-Za-z0-9,\\-]{1,24}$).\n"
        "\n"
        "SCHEMA (return ONE JSON object only):\n"
        "{\n"
        '  "id": "<date_key>",\n'
        '  "title": str,\n'
        '  "intro": str,\n'
        '  "npc_lines": [str]?,\n'
        '  "supplies_start": 3,\n'
        '  "rooms": [\n"'
        '    { "id": "room_1", "title": str, "text": str,\n'
        '      "routes": [\n'
        '        { "id": "cautious", "label": "Proceed carefully",\n'
        '          "puzzle": { "id": "r1_caut", "type": "mini", "mechanic": "multiple_choice|sequence_input|grid_input|text_input",\n'
        '                      "ui_spec": object, "prompt": str,\n'
        '                      "answer_format": {"pattern": str}, "solution": {"answer": str}, "hints": [str, str]? } },\n'
        '        { "id": "brisk", "label": "Move quickly", "puzzle": { ... type: "mini" ... } },\n'
        '        { "id": "risky", "label": "Take a risk", "puzzle": { ... type: "mini" ... } }\n'
        '      ],\n'
        '      "fragment_rule": "FIRST2|LAST2|FIRST3|LAST3|IDX:i,j"\n'
        '    },\n'
        '    { "id": "room_2", ... },\n'
        '    { "id": "room_3", ... }\n'
        '  ],\n'
        '  "final": { "id": "final", "prompt": "Assemble the three fragments to form the PASSCODE.",\n'
        '             "answer_format": {"pattern": "^[A-Za-z0-9]{4,12}$"}, "solution": {"answer": "<concat>"} },\n'
        '  "difficulty": "medium"\n'
        "}\n"
        "\n"
        f"DATE_KEY: {date_key}\n"
    )


def _extract_json_block(text: str) -> Optional[str]:
    if not text or not text.strip(): return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t); t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    return t[i:j+1] if i!=-1 and j!=-1 and j>i else None

def llm_generate_trailroom(date_key: str) -> Optional[Dict[str, Any]]:
    client, mode = _get_openai_client()
    if not client:
        current_app.logger.warning("[escape] LLM: no client (missing API key or FORCE_OFFLINE).")
        return None

    sys = "You are a careful game designer. Return ONE JSON object only. Follow the schema exactly. No prose."
    user = _trail_prompt(date_key)
    model = os.getenv("ESCAPE_MODEL", "gpt-5-chat-latest")
    temp  = float(os.getenv("ESCAPE_TEMP", "0.9"))
    top_p = float(os.getenv("ESCAPE_TOP_P", "0.95"))

    def _parse(text: str) -> Optional[Dict[str, Any]]:
        jb = _extract_json_block(text) or text
        try:
            return json.loads(jb) if jb else None
        except Exception:
            current_app.logger.warning("[escape] LLM: JSON parse failed. head=%r", (text or "")[:200])
            return None

    try:
        if mode == "modern":
            # --- Try Responses API (messages form)
            try:
                resp = client.responses.create(
                    model=model,
                    messages=[{"role":"system","content":sys}, {"role":"user","content":user}],
                    temperature=temp, top_p=top_p,
                )
                text = getattr(resp, "output_text", None)
                if not text:
                    try: text = resp.output[0].content[0].text
                    except Exception: text = ""
                jr = _parse(text)
                if jr:
                    current_app.logger.info("[escape] LLM: responses(messages) -> OK")
                    return jr
            except TypeError as te:
                current_app.logger.info("[escape] LLM: responses(messages) signature mismatch: %s", te)
            except Exception as e:
                current_app.logger.exception("[escape] LLM: responses(messages) failed: %s", e)

            # --- Try Responses API (single string input)
            try:
                resp = client.responses.create(
                    model=model,
                    input=sys + "\n\n" + user,
                    temperature=temp, top_p=top_p,
                )
                text = getattr(resp, "output_text", "") or ""
                jr = _parse(text)
                if jr:
                    current_app.logger.info("[escape] LLM: responses(input=str) -> OK")
                    return jr
            except Exception as e:
                current_app.logger.exception("[escape] LLM: responses(input=str) failed: %s", e)

            # --- Try Chat Completions (no response_format!)
            try:
                cc = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user",   "content": user}],
                    temperature=temp, top_p=top_p,
                )
                text = (cc.choices[0].message.content or "").strip()
                jr = _parse(text)
                if jr:
                    current_app.logger.info("[escape] LLM: chat.completions -> OK")
                    return jr
            except Exception as e:
                current_app.logger.exception("[escape] LLM: chat.completions failed: %s", e)

        else:
            # legacy
            text = client.ChatCompletion.create(  # type: ignore
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user",   "content": user}],
                temperature=temp,
            )["choices"][0]["message"]["content"]
            jr = _parse(text)
            if jr:
                current_app.logger.info("[escape] LLM: legacy ChatCompletion -> OK")
            return jr

    except Exception as e:
        current_app.logger.exception("[escape] llm_generate_trailroom outer failed: %s", e)

    current_app.logger.warning("[escape] LLM: no usable result; falling back to offline.")
    return None

# ───────────────────────── Critic (safe patch) ─────────────────────────

def llm_critic_patch(room_json: Dict[str, Any]) -> Dict[str, Any]:
    client, mode = _get_openai_client()
    if not client:
        return room_json

    content = (
        "You are a flavor-only editor. Improve novelty/tone of 'title', 'intro', "
        "'npc_lines' (if present), and 'final.prompt'. Return ONLY a JSON array of "
        "RFC6902 JSON-Patch ops. If no changes, return [].\n\n"
        f"ROOM_JSON:\n{json.dumps(room_json, ensure_ascii=False)}"
    )

    try:
        model = os.getenv("ESCAPE_MODEL", "gpt-5-mini")  # any Responses-capable model
        text = "[]"
        if mode == "modern":
            try:
                r = client.responses.create(model=model, input=content, temperature=0.0)
                text = getattr(r, "output_text", "[]") or "[]"
            except Exception:
                # last resort: chat without response_format
                c = client.chat.completions.create(
                    model=model, messages=[{"role":"user","content":content}], temperature=0.0
                )
                text = (c.choices[0].message.content or "[]")
        else:
            c = client.ChatCompletion.create(  # type: ignore
                model=model, messages=[{"role":"user","content":content}], temperature=0.0
            )
            text = c["choices"][0]["message"]["content"] or "[]"

        jb = _extract_json_block(text) or text
        ops = json.loads(jb)
        if not isinstance(ops, list):
            return room_json

        # Apply very narrow patch
        def _allowed(parts):
            if not parts: return False
            if parts[0] in {"title","intro","npc_lines"}: return True
            return parts[0]=="final" and len(parts)>1 and parts[1]=="prompt"

        blob = json.loads(json.dumps(room_json))
        for op in ops:
            path = op.get("path","")
            parts = [p for p in path.split("/") if p]
            if not _allowed(parts): continue
            if op.get("op") in {"replace","add"}:
                cur = blob
                for i,p in enumerate(parts):
                    if i == len(parts)-1:
                        cur[p] = op.get("value")
                    else:
                        cur = cur.setdefault(p, {})
        return blob
    except Exception:
        return room_json

# ───────────────────────── Validation / assembly ─────────────────────────

def _recent_answer_set(days: int = ANSWER_COOLDOWN_DAYS) -> set:
    """Collect normalized answers seen in the cooldown window (both flat and trail shapes)."""
    db, EscapeRoom = _get_db_and_models()
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    S = set()
    rows = db.session.query(EscapeRoom).filter(EscapeRoom.created_at >= cutoff).all()
    for r in rows:
        blob = r.json_blob or {}
        # flat shape
        for p in _as_dict_list(blob.get("puzzles")):
            sol = (p.get("solution") or {}).get("answer")
            if sol:
                S.add(normalize_answer(sol))
        # trail shape
        for rm in _as_dict_list(blob.get("trail", {}).get("rooms")):
            for rt in _as_dict_list(rm.get("routes")):
                pp = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
                sol = (pp.get("solution") or {}).get("answer")
                if sol:
                    S.add(normalize_answer(sol))
    return S

def _validate_minigame(p: Dict[str, Any]) -> None:
    # Required base fields
    for k in ("id","type","prompt","answer_format","solution"):
        if k not in p:
            raise ValueError(f"Mini-game missing '{k}'")
    if p["type"] != "mini":
        raise ValueError("Mini-game must have type='mini'")

    mech = (p.get("mechanic") or "").strip()
    if mech not in ALLOWED_MECHANICS:
        raise ValueError(f"Mini-game mechanic not allowed: {mech}")

# AFTER
    ans = (p.get("solution") or {}).get("answer")
    # If LLM emitted a list for sequence_input, join it.
    if (p.get("mechanic") == "sequence_input") and isinstance(ans, list):
        ans = ",".join(str(t).strip() for t in ans if str(t).strip())
        p.setdefault("solution", {})["answer"] = ans
    if not ans or not normalize_answer(ans):
        raise ValueError("Mini-game has empty solution")

    # UI spec sanity per mechanic
    ui = p.get("ui_spec") or {}
    if mech == "multiple_choice":
        opts = ui.get("options")
        if not (isinstance(opts, list) and len(opts) >= 4):
            raise ValueError("multiple_choice requires ≥4 options")
        if _looks_trivial_multiple_choice(p):
            raise ValueError("multiple_choice puzzle too trivial")
        if not isinstance(opts, list) or len(opts) < 2:
            raise ValueError("multiple_choice requires >=2 options in ui_spec.options")
        if str(ans) not in [str(x) for x in opts]:
            # We accept normalized match too
            if normalize_answer(str(ans)) not in {normalize_answer(str(x)) for x in opts}:
                raise ValueError("answer must be one of options for multiple_choice")
    elif mech == "sequence_input":
        seq = ui.get("sequence")
        cue_ok = bool(ui.get("cue_set") or ui.get("cues_audio") or ui.get("cues"))
        if not ((isinstance(seq, list) and seq) or cue_ok):
            raise ValueError("sequence_input requires non-empty ui_spec.sequence or cue_set/cues_audio")
        steps = [t for t in re.split(r"[,\s]+", str(ans)) if t]
        if len(steps) < 5:
            raise ValueError("sequence_input answer must be at least 5 steps")

    elif mech == "grid_input":
        grid = ui.get("grid")
        if not (isinstance(grid, list) and 2 <= len(grid) <= 6 and all(isinstance(r, list) for r in grid)):
            raise ValueError("grid_input requires ui_spec.grid as 2..6 rows")
        if len(str(ans)) < 4:
            raise ValueError("grid_input answer must require at least 4 taps")
        # NEW: if the prompt talks about letters/words, demand single-char cells
        if re.search(r"\b(letter|letters|spell|word|collect)\b", (p.get("prompt") or ""), re.I):
            if any(len(str(cell)) > 1 for row in grid for cell in row):
                raise ValueError("grid_input uses multi-char cells for a letter-collection puzzle")

    # Pattern must be anchored and match answer (normalized accepted)
    pat = ((p.get("answer_format") or {}).get("pattern")) or _default_pattern_for_answer(ans)
    if not re.fullmatch(r"^\^.*\$$", pat):
        raise ValueError("answer_format.pattern must be anchored ^...$")
    if not (re.fullmatch(pat, str(ans)) or re.fullmatch(pat, normalize_answer(str(ans)))):
        raise ValueError("Mini-game solution does not match declared pattern")

    # Stock/cooldown protection
    if normalize_answer(ans).lower() in {s.lower() for s in COMMON_STOCK_ANSWERS}:
        raise ValueError("Mini-game uses a stock/cliché answer")
    if answer_recently_used(str(ans)):
        raise ValueError("Mini-game answer recently used")

def _validate_puzzle(p: Dict[str, Any]) -> None:
    # Mini-games use their own validator
    if (p.get("type") or "").lower() == "mini":
        _validate_minigame(p)
        return

    # --- existing legacy validation below (unchanged) ---
    for k in ("id","prompt","answer_format","solution"):
        if k not in p: raise ValueError(f"Puzzle missing '{k}'")
    if p.get("type") not in ALLOWED_TYPES:
        raise ValueError("Puzzle type not allowed")
    ans = (p.get("solution") or {}).get("answer")
    if not ans or not normalize_answer(ans):
        raise ValueError("Empty solution")
    if normalize_answer(ans).lower() in {s.lower() for s in COMMON_STOCK_ANSWERS}:
        raise ValueError("Puzzle uses stock/cliché answer")
    pattern = (p.get("answer_format") or {}).get("pattern")
    if not pattern or not re.match(r"^\^.*\$$", pattern):
        raise ValueError("answer_format.pattern must be anchored ^...$")
    if not (re.fullmatch(pattern, str(ans)) or re.fullmatch(pattern, normalize_answer(str(ans)))):
        raise ValueError("Solution does not match declared pattern")
    if answer_recently_used(str(ans)):
        raise ValueError("Answer recently used")

def _coerce_same_fragment_or_const(rule: str, p1: Dict[str,Any], p2: Dict[str,Any]) -> Tuple[str,str]:
    if not is_valid_fragment_rule(rule):
        rule = "FIRST2"
    a1 = (p1.get("solution") or {}).get("answer","")
    a2 = (p2.get("solution") or {}).get("answer","")
    f1 = apply_fragment_rule(a1, rule)
    f2 = apply_fragment_rule(a2, rule)
    if f1 != f2:
        rule = f"CONST:{f1}"
        f2 = f1
    return rule, f1

def _flatten_puzzles(trail: Dict[str,Any]) -> List[Dict[str,Any]]:
    out=[]
    for rm in _as_dict_list(trail.get("rooms")):
        for rt in _as_dict_list(rm.get("routes")):
            p = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
            if p:
                pj = json.loads(json.dumps(p))
                pj.setdefault("archetype", pj.get("type"))
                pj["room_id"] = rm.get("id")
                pj["route_id"] = rt.get("id")
                # make sure mini meta survives
                if pj.get("type") == "mini":
                    pj["mechanic"] = pj.get("mechanic")
                    pj["ui_spec"]  = pj.get("ui_spec") or {}
                out.append(pj)
    return out

def validate_trailroom(room: Dict[str,Any]) -> Dict[str,Any]:
    # required top-level
    for k in ("id","title","intro","trail","final","difficulty"):
        if k not in room: raise ValueError(f"Room missing '{k}'")

    # sanitize trail structure
    trail = room["trail"] if isinstance(room["trail"], dict) else {}
    rooms = _as_dict_list(trail.get("rooms"))
    if len(rooms) != 3:
        raise ValueError("Trail must contain 3 rooms")
    trail["rooms"] = rooms
    room["trail"] = trail

    # puzzles validation + consistent fragment per room
    # puzzles validation + consistent fragment per room
    fragments = []
    for rm in rooms:
        routes = _as_dict_list(rm.get("routes"))
        if len(routes) != 3:
            raise ValueError("Each room must have exactly 3 routes")
        pz = []
        for r in routes:
            q = r.get("puzzle") if isinstance(r.get("puzzle"), dict) else {}
            _validate_puzzle(q)
            pz.append(q)
        fr_rule = rm.get("fragment_rule") or "FIRST2"
        fr_rule, frag = _coerce_same_fragment_or_const_all(fr_rule, pz)
        rm["routes"] = routes
        rm["fragment_rule"] = fr_rule
        fragments.append(frag)

    # final answer: must equal concat of fragments (trim to bounds)
    final = room["final"] if isinstance(room["final"], dict) else {}
    expect = (final.get("solution") or {}).get("answer","")
    concat = "".join(fragments)
    concat_norm = normalize_answer(concat)[:FINAL_CODE_MAX_LEN]
    if len(concat_norm) < FINAL_CODE_MIN_LEN:
        concat_norm = (concat_norm + "X"*FINAL_CODE_MIN_LEN)[:FINAL_CODE_MAX_LEN]
    if normalize_answer(expect) != concat_norm:
        final.setdefault("solution", {})["answer"] = concat_norm
    room["final"] = final

    # attach flat puzzles for legacy UI/logic
    room["puzzles"] = _flatten_puzzles(room["trail"])

    # expose final_code for UI compatibility
    room["final_code"] = (room.get("final", {}).get("solution", {}) or {}).get("answer")

    # anti_spoiler (defaults)
    room.setdefault("anti_spoiler", {"paraphrase_variants": 3, "decoys": 2})
    return room

# Old-shape validator kept for backwards compatibility (not used in trail flow)
def validate_room(room_json: Dict[str, Any]) -> Dict[str, Any]:
    required_top = {"id", "title", "intro", "graph", "puzzles", "anti_spoiler", "difficulty"}
    missing = [k for k in required_top if k not in room_json]
    if missing:
        raise ValueError(f"Room missing keys: {missing}")

    raw_puzzles = room_json.get("puzzles", [])
    if not isinstance(raw_puzzles, list):
        raise ValueError("Room puzzles must be a list")
    puzzles: List[Dict[str, Any]] = [p for p in raw_puzzles if isinstance(p, dict)]
    if not puzzles:
        raise ValueError("Room must contain at least one puzzle")
    room_json["puzzles"] = puzzles

    seen_ids = set()
    for p in puzzles:
        for key in ("id", "archetype", "prompt", "answer_format", "solution"):
            if key not in p:
                raise ValueError(f"Puzzle missing '{key}'")
        pid = p["id"]
        if pid in seen_ids:
            raise ValueError("Duplicate puzzle id")
        seen_ids.add(pid)

        sol = p.get("solution", {}) or {}
        ans = sol.get("answer")
        if not ans or len(normalize_answer(ans)) == 0:
            raise ValueError("Puzzle has empty solution answer")

        if normalize_answer(ans).lower() in {s.upper() for s in COMMON_STOCK_ANSWERS}:
            raise ValueError("Puzzle uses stock/cliché answer")

        if answer_recently_used(str(ans)):
            raise ValueError("Answer recently used in the cooldown window")

        pattern = (p.get("answer_format") or {}).get("pattern")
        if pattern:
            if not re.match(r"^\^.*\$$", pattern):
                raise ValueError("answer_format.pattern must be anchored ^...$")
            if not re.match(pattern, str(ans)):
                if not re.match(pattern, normalize_answer(str(ans))):
                    raise ValueError("Solution does not match its declared pattern")

    graph = room_json.get("graph", {}) or {}
    if "end" not in graph:
        raise ValueError("Graph must define an 'end' gate id")

    fc = room_json.get("final_code")
    if fc:
        if not (FINAL_CODE_MIN_LEN <= len(normalize_answer(fc)) <= FINAL_CODE_MAX_LEN):
            raise ValueError("final_code length out of bounds")

    return room_json

def too_easy(room: Dict[str,Any]) -> bool:
    puzzles = room.get("puzzles") or []
    if len(puzzles) <= 1: return True
    easy=0
    for p in puzzles:
        arch = p.get("type") or p.get("archetype")
        ans = normalize_answer((p.get("solution") or {}).get("answer",""))
        pattern = (p.get("answer_format") or {}).get("pattern","")
        if arch in {"anagram","caesar"} and 4 <= len(ans) <= 6: easy += 1
        if arch == "numeric_lock" and pattern == r"^\d{4}$":     easy += 1
        if ans in {"1234","0000","1111","2580"}:                 easy += 1
    return easy >= len(puzzles)

def harden(room: Dict[str,Any], rng: Optional[random.Random]=None) -> Dict[str,Any]:
    rng = rng or random.Random()
    for p in (room.get("puzzles") or []):
        decs = p.get("decoys") or []; ans = str((p.get("solution") or {}).get("answer",""))
        while len(decs) < 3:
            if (p.get("type") in {"anagram","caesar","vigenere"} or p.get("archetype") in {"anagram","caesar","vigenere"}) and ans.isalpha():
                ls=list(ans); rng.shuffle(ls); decs.append("".join(ls))
            elif re.fullmatch(r"^\d+$", ans):
                num=int(ans); tweak=(num+rng.randrange(1,9))%(10**len(ans)); decs.append(str(tweak).zfill(len(ans)))
            else: decs.append(ans[::-1])
        p["decoys"] = decs
    room["difficulty"] = room.get("difficulty") or "hard"
    anti = room.get("anti_spoiler", {}); anti["decoys"] = max(3, anti.get("decoys", 2)); room["anti_spoiler"] = anti
    return room

# ───────────────────────── Offline fallback ─────────────────────────

def _offline_trail(date_key: str, rng: random.Random) -> Dict[str, Any]:
    title, intro = rng.choice(THEMES)

    recent_norm = _recent_answer_set()
    blacklist = {s.lower() for s in recent_norm} | {s.lower() for s in COMMON_STOCK_ANSWERS}

    rooms = []
    fragments = []
    for i in range(1, 4):
        theme = ["Glass Arcade","Switchyard","Ferry Ramp"][i-1] if i<=3 else f"Stop {i}"
        kinds = rng.sample(["acrostic","tapcode","pathcode"], k=3)
        gens = {
            "acrostic": gen_acrostic,
            "tapcode": gen_tapcode,
            "pathcode": gen_pathcode
        }
        p1 = gens[kinds[0]](rng, f"r{i}_caut_pz",  blacklist, theme).to_json()
        p2 = gens[kinds[1]](rng, f"r{i}_brisk_pz", blacklist, theme).to_json()
        p3 = gens[kinds[2]](rng, f"r{i}_risky_pz", blacklist, theme).to_json()

        fr, frag = _coerce_same_fragment_or_const_all("FIRST2", [p1, p2, p3])
        fragments.append(frag)
        rooms.append({
            "id": f"room_{i}",
            "title": theme,
            "text": ["LEDs stutter on a damp console.","Signals tick from a rusted panel.","Fog beads across the ticket glass."][i-1] if i<=3 else "",
            "routes": [
                {"id":"cautious","label":"Proceed carefully","puzzle":p1},
                {"id":"brisk","label":"Move quickly","puzzle":p2},
                {"id":"risky","label":"Take a risk","puzzle":p3},
            ],
            "fragment_rule": fr
        })

    concat = "".join(fragments)
    final = normalize_answer(concat)[:FINAL_CODE_MAX_LEN]
    if len(final) < FINAL_CODE_MIN_LEN:
        final = (final + "X"*FINAL_CODE_MIN_LEN)[:FINAL_CODE_MAX_LEN]

    trail = {"supplies_start": SUPPLIES_START_DEFAULT, "rooms": rooms}
    room = {
        "id": date_key, "title": title, "intro": intro,
        "npc_lines": [], "difficulty": DEFAULT_DIFFICULTY,
        "trail": trail,
        "final": {"id":"final", "prompt":"Assemble the three fragments to form the PASSCODE.",
        "answer_format":{"pattern": r"^[A-Za-z0-9]{4,12}$"},
                  "solution":{"answer": final}},
        "anti_spoiler": {"paraphrase_variants": 3, "decoys": 2},
    }
    room = _reshuffle_mechanics_for_variety(room)


    # NEW: avoid cooldown collisions in offline flow
    try:
        room = _replace_recent_answers(room, rng)
    except Exception as e:
        try: current_app.logger.warning("[escape] offline replace_recent skipped: %s", e)
        except Exception: pass

    # 🔧 Make sure minis are self-consistent before validation (offline flow too)
    try:
        room = _fixup_minigames(room, rng)
    except Exception as e:
        try: current_app.logger.warning("[escape] offline mini fixup skipped: %s", e)
        except Exception: pass

    room = validate_trailroom(room)
    return harden(room, rng)

def generate_room_offline(date_key: str, server_secret: str) -> Dict[str, Any]:
    salt = os.getenv("ESCAPE_REGEN_SALT", "")
    seed = daily_seed(date_key, server_secret + salt)   # ← was server_secret only
    rng = rng_from_seed(seed)
    room = _offline_trail(date_key, rng)
    room["source"] = "offline"
    return room

# ───────────────────────── Primary generation ─────────────────────────

# core.py

def _upgrade_minigame_hints(p: Dict[str, Any]) -> None:
    if (p.get("type") != "mini"):  # only for mini-games
        return
    hints = [h for h in (p.get("hints") or []) if isinstance(h, str) and h.strip()]
    mech  = (p.get("mechanic") or "").strip()
    ui    = p.get("ui_spec") or {}
    ans   = str((p.get("solution") or {}).get("answer", ""))

    # Mechanic-specific add-ons
    extra: List[str] = []
    if mech == "sequence_input":
        ui = p.get("ui_spec") or {}
        ans = str((p.get("solution") or {}).get("answer", ""))
        if isinstance(ui.get("cues_audio"), list) and ui["cues_audio"]:
            steps_len = len(ui["cues_audio"])
        else:
            steps_len = len([t for t in str(ans).split(",") if t.strip()])
        if steps_len:
            extra.append(f"Sequence length: {steps_len}.")
        extra.append("Use only the chips above; order matters.")
    elif mech == "multiple_choice":
        opts = ui.get("options")
        if not (isinstance(opts, list) and len(opts) >= 4):
            raise ValueError("multiple_choice requires ≥4 options")
        if _looks_trivial_multiple_choice(p):
            raise ValueError("multiple_choice puzzle too trivial")

        # Support strings or dicts with {id,label,...}
        norm = _normalize_mc_options(opts)
        labels = [o["label"] for o in norm]
        ids    = [o["id"]    for o in norm]
        ok = (
            str(ans) in labels
            or normalize_answer(str(ans)) in {normalize_answer(s) for s in labels}
            or str(ans) in ids
            or normalize_answer(str(ans)) in {normalize_answer(s) for s in ids}
        )
        if not ok:
            raise ValueError("answer must be one of options for multiple_choice")

        # Basic sprite-pack shape is allowed (if present)
        if "shadow_rule" in ui:
            sr = ui["shadow_rule"]
            if not isinstance(sr, dict) or "target" not in sr or "lengths" not in sr:
                raise ValueError("ui_spec.shadow_rule must include 'target' and 'lengths'")

    # Keep the first two good hints
    p["hints"] = (hints + [h for h in extra if h not in hints])[:2]

def gen_signal_translate(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    """
    Audio memory: play a sequence of cues (3 distinct kinds). Player taps the same
    cues in order. No legend; the cues ARE the inputs.
    """
    # Full bank (kept for variety)
    cue_bank = ["short beep", "long beep", "hiss", "clank", "drip", "gust", "rumble", "chime"]

    # Use exactly 3 distinct cues to keep the surface simple
    cues = rng.sample(cue_bank, 3)

    # Build a 7–9 step sequence using ONLY those 3 cues
    k = rng.randrange(7, 10)
    seq = [rng.choice(cues) for _ in range(k)]

    # Decoys
    rev   = list(reversed(seq))
    rot   = seq[1:] + seq[:1]
    tweak = seq[:]
    ti = rng.randrange(0, k)
    tweak[ti] = rng.choice([c for c in cues if c != seq[ti]])

    prompt = (
        f"From the {theme or 'hall'}, signals repeat. Listen and reproduce the exact cue "
        "sequence by tapping the cue bubbles. Use the ▶ Play button to hear them again."
    )

    return Puzzle(
        id=pid,
        archetype="mini",
        prompt=prompt,
        answer_format={"pattern": r"^[A-Za-z ,\-]{5,200}$"},
        solution={"answer": ",".join(seq)},
        hints=[f"Sequence length: {k}.", "Each tap appends a cue; use Undo/Clear to fix mistakes."],
        decoys=[",".join(rev), ",".join(rot), ",".join(tweak)],
        paraphrases=[
            "A repeating pattern of sounds—match them in order.",
            "Replay the audio and tap the same cues."
        ],
        mechanic="sequence_input",
        # Only show the 3 buttons needed for this round
        ui_spec={"cue_set": cues, "cues_audio": seq},
    )

# NEW: last-mile hardening so minis can't fail validation due to pattern/answer drift
def _force_pattern_match(p: Dict[str, Any]) -> None:
    ans = (p.get("solution") or {}).get("answer", "")
    mech = (p.get("mechanic") or "").strip()

    # If the LLM emitted a list for sequence_input, join it now.
    if mech == "sequence_input" and isinstance(ans, list):
        ans = ",".join(str(t).strip() for t in ans if str(t).strip())
        p.setdefault("solution", {})["answer"] = ans

    need = _default_pattern_for_answer(str(ans))
    af = p.get("answer_format") or {}
    pat = (af.get("pattern") or need).strip()

    # If unanchored or mismatch, coerce to a pattern that matches the answer we actually store.
    if (not re.fullmatch(r"^\^.*\$$", pat)) or not (
        re.fullmatch(pat, str(ans)) or re.fullmatch(pat, normalize_answer(str(ans)))
    ):
        af["pattern"] = need
    p["answer_format"] = af

def _ensure_label_path_playable(rng: random.Random, p: Dict[str, Any]) -> None:
    """If answer is a sequence of multi-char grid labels, make sure the grid
    contains them in order on an adjacent path; rebuild placement if not."""
    if (p.get("mechanic") != "grid_input"): return
    ui = p.get("ui_spec") or {}
    grid = ui.get("grid")
    if not (isinstance(grid, list) and grid and all(isinstance(r, list) for r in grid)): return

    labels_flat = [str(cell) for row in grid for cell in row]
    # Extract the intended label path from the solution using existing labels
    ans = str((p.get("solution") or {}).get("answer", ""))
    if not ans: return
    lab_re = re.compile("|".join(sorted(map(re.escape, set(labels_flat)), key=len, reverse=True)))
    path = lab_re.findall(ans)
    if len(path) < 2: return  # nothing to fix

    # Map labels to coords (first occurrence)
    coords = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            coords.setdefault(str(cell), (r, c))

    def adjacent(a, b):
        (r1, c1), (r2, c2) = coords.get(a, (-9, -9)), coords.get(b, (9, 9))
        return abs(r1 - r2) + abs(c1 - c2) == 1

    have_all = all(x in coords for x in path)
    ok = have_all and all(adjacent(path[i], path[i+1]) for i in range(len(path)-1))

    if ok:
        # Make sure prompt reflects correct count/start/end
        taps = len(path)
        clar = f"\nStart at {path[0]}; end at {path[-1]}. Move only to adjacent tiles (no diagonals). Exactly {taps} taps."
        if clar.strip() not in (p.get("prompt") or ""):
            p["prompt"] = (p.get("prompt","").rstrip() + clar).strip()
        return

    # Rebuild: place the labels along a snake path to guarantee adjacency
    n_rows, n_cols = len(grid), len(grid[0])
    snake = []
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else range(n_cols-1, -1, -1)
        for c in cols:
            snake.append((r, c))
    # Place the path on the first len(path) cells of the snake
    new_grid = [row[:] for row in grid]
    for (lab, (r, c)) in zip(path, snake[:len(path)]):
        new_grid[r][c] = lab
    ui["grid"] = new_grid
    p["ui_spec"] = ui

    taps = len(path)
    clar = f"\nStart at {path[0]}; end at {path[-1]}. Move only to adjacent tiles (no diagonals). Exactly {taps} taps."
    if clar.strip() not in (p.get("prompt") or ""):
        p["prompt"] = (p.get("prompt","").rstrip() + clar).strip()

def _fixup_minigames(blob: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    trail = blob.get("trail") or {}
    rooms = trail.get("rooms") or []
    for rm in rooms:
        theme = rm.get("title","") or rm.get("text","")
        for rt in (rm.get("routes") or []):
            p = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
            if (p.get("type") or p.get("archetype")) != "mini":
                continue

            pid = p.get("id") or f"{rm.get('id','room')}_{rt.get('id','route')}_pz"
            mech = (p.get("mechanic") or "").strip().lower()
            ui = p.setdefault("ui_spec", {})

            # Ensure valid mechanic
            if mech not in ALLOWED_MECHANICS:
                p["mechanic"] = "text_input"
                mech = "text_input"

            # Sequence: guarantee tokens, join lists, enforce length ≥5
            if mech == "sequence_input":
                if not ui.get("sequence"):
                    ui["sequence"] = ["tap","hold","left","right","up","down","rotate_left","rotate_right"]

                # Join list-style answers
                ans = (p.get("solution") or {}).get("answer", "")
                if isinstance(ans, list):
                    ans = ",".join(str(t).strip() for t in ans if str(t).strip())
                    p.setdefault("solution", {})["answer"] = ans

                steps = [t for t in re.split(r"[,\s]+", str(ans)) if t]

                # Detect if the prompt actually exposes a series/cues (not just our injected "Controls:")
                def _has_explicit_series(txt: str, tokens: List[str]) -> bool:
                    t = re.sub(r"(?is)controls:\s*[-•].*$", "", txt or "")  # strip any appended legend
                    token_pat = r"(?:%s)" % "|".join(re.escape(x) for x in tokens)
                    # either digit pairs like 5-4, 2-3 ... or 3+ comma-separated action tokens
                    return bool(re.search(r"\b\d\s*-\s*\d\b", t)) or bool(re.search(
                        rf"\b{token_pat}\b(?:\s*,\s*\b{token_pat}\b){{2,}}", t))

                has_series = _has_explicit_series(p.get("prompt",""), ui["sequence"])

                # If too short OR doesn't reveal any cues, swap in a guaranteed-playable generator
                if (len(steps) < 5) or (not has_series):
                    rt["puzzle"] = gen_signal_translate(rng, pid, set(), theme).to_json()
                    p  = rt["puzzle"]
                    mech = (p.get("mechanic") or "").strip().lower()
                    ui = p.setdefault("ui_spec", {})

            # Grid: if invalid or uses multi-char cells for a "letters/word" prompt, fix or swap
            if mech == "grid_input":
                grid = ui.get("grid")
                bad_grid = not (isinstance(grid, list) and grid and all(isinstance(r, list) for r in grid))
                prompt_txt = p.get("prompt") or ""
                needs_letters = bool(re.search(r"\b(letter|letters|spell|word|collect)\b", prompt_txt, re.I))
                long_tokens = any(len(str(cell)) > 1 for row in (grid or []) for cell in row) if not bad_grid else False
                sol = (p.get("solution") or {}).get("answer", "")

                if bad_grid or (needs_letters and long_tokens):
                    if re.fullmatch(r"^[A-Za-z]{4,12}$", re.sub(r"[^A-Za-z]", "", str(sol)) or ""):
                        _force_letter_grid_for_answer(rng, p, sol)
                        p["prompt"] = (prompt_txt.rstrip() + "\nTap tiles to collect letters that spell the word.").strip()
                    else:
                        rt["puzzle"] = gen_knightword(rng, pid, set(), theme).to_json()
                        p = rt["puzzle"]
                        mech = (p.get("mechanic") or "").strip().lower()
                        ui = p.setdefault("ui_spec", {})

                # NEW: enforce >=4 taps even if grid is otherwise valid
                if len(re.sub(r"[^A-Za-z0-9]", "", str(sol) or "")) < 4:
                    rt["puzzle"] = gen_knightword(rng, pid, set(), theme).to_json()
                    p = rt["puzzle"]
                    mech = (p.get("mechanic") or "").strip().lower()
                    ui = p.setdefault("ui_spec", {})

            # Clarify ambiguous grid paths
            _ensure_label_path_playable(rng, p)
            _annotate_grid_path_meta(p)

            # NEW: last-mile reroll if the answer is on cooldown
            try:
                p = _reroll_recent_mini(rng, p, set(), theme)
            except Exception as e:
                try: current_app.logger.warning("[escape] reroll_recent_mini (fixup) skipped: %s", e)
                except Exception: pass

            # Finally, force the pattern and hints
            _force_pattern_match(p)
            _upgrade_minigame_hints(p)
            rt["puzzle"] = p

            # Finally, force the pattern and hints
            _force_pattern_match(p)
            _upgrade_minigame_hints(p)

            # write back in case we mutated
            rt["puzzle"] = p
    return blob

def _classify_mini_kind(p: Dict[str, Any]) -> str:
    """
    Map a mini into one of 6 flavors we want daily:
    'signal' (cue-only sequence), 'legend' (sequence with legend),
    'pathcode' (grid path), 'knightword' (grid knight hops),
    'tapcode' (Polybius 5x5), 'acrostic' (first-letters word).
    Returns '' if unknown.
    """
    if (p.get("type") or p.get("archetype")) != "mini":
        return ""
    mech = (p.get("mechanic") or "").lower()
    ui   = p.get("ui_spec") or {}
    prompt = (p.get("prompt") or "")
    sol    = (p.get("solution") or {})

    # sequence flavors
    if mech == "sequence_input":
        if ui.get("cue_set") or ui.get("cues_audio") or ui.get("cues"):
            return "signal"
        if "legend" in prompt.lower() or "legend" in json.dumps(sol).lower():
            return "legend"
        return "signal"  # default ambiguous sequences into signal

    # grid flavors
    if mech == "grid_input":
        if sol.get("rule") == "knight" or re.search(r"\bknight\b", prompt, re.I) or \
           re.search(r"\bknight\b", (ui.get("notes") or ""), re.I):
            return "knightword"
        if sol.get("path") or re.search(r"\bpath\b|follow the path|directions", prompt, re.I):
            return "pathcode"

    # text flavors
    if re.search(r"\bpolybius\b|5×5|5x5|I/J share", prompt, re.I):
        return "tapcode"
    if re.search(r"\b(first letters|acrostic)\b", prompt, re.I):
        return "acrostic"

    return ""

def _enforce_daily_mini_variety(blob: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    Ensure the 9 minis (3 rooms × 3 routes) cover all 6 flavors at least once.
    If a flavor is missing, replace an overrepresented mini with a generated one of that flavor.
    """
    wanted = ["signal","legend","pathcode","knightword","tapcode","acrostic"]

    # gather minis
    rooms = (blob.get("trail") or {}).get("rooms") or []
    catalog = []  # (room_idx, route_idx, puzzle, kind)
    for i, rm in enumerate(rooms):
        for j, rt in enumerate(rm.get("routes") or []):
            p = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
            kind = _classify_mini_kind(p)
            catalog.append((i, j, p, kind))

    from collections import Counter
    counts = Counter(k for *_, k in catalog if k)

    # quick exit if all covered
    if all(counts.get(k, 0) > 0 for k in wanted):
        return blob

    # helper: fabricate a new puzzle of given flavor
    def _make(kind: str, pid: str, theme: str) -> Dict[str, Any]:
        if   kind == "signal":     q = gen_signal_translate(rng, pid, set(), theme)
        elif kind == "legend":     q = gen_translate_with_legend(rng, pid, set(), theme)
        elif kind == "pathcode":   q = gen_pathcode(rng, pid, set(), theme)
        elif kind == "knightword": q = gen_knightword(rng, pid, set(), theme)
        elif kind == "tapcode":    q = gen_tapcode(rng, pid, set(), theme)
        else:                      q = gen_acrostic(rng, pid, set(), theme)
        return q.to_json()

    # find overrepresented victims (prefer the flavor with the highest count)
    def _pick_victim():
        # pick any entry whose kind currently has the max count
        if not catalog:
            return None
        # compute live counts each time
        live_counts = Counter(k for *_, k in catalog if k)
        if not live_counts:
            return None
        worst_kind, _ = max(live_counts.items(), key=lambda kv: kv[1])
        # choose a concrete puzzle of that kind
        choices = [(idx, entry) for idx, entry in enumerate(catalog) if entry[3] == worst_kind]
        return rng.choice(choices) if choices else None

    # replace until all flavors exist or we run out of safe victims
    for missing in [k for k in wanted if counts.get(k, 0) == 0]:
        victim = _pick_victim()
        if not victim:
            break
        v_idx, (ri, rj, old_p, old_kind) = victim
        theme = (rooms[ri].get("title") or rooms[ri].get("text") or "")
        pid   = (old_p.get("id") if isinstance(old_p, dict) else f"r{ri+1}_{rj}_pz") or f"r{ri+1}_{rj}_pz"
        new_p = _make(missing, pid, theme)

        # write back
        rooms[ri]["routes"][rj]["puzzle"] = new_p
        catalog[v_idx] = (ri, rj, new_p, missing)
        counts[old_kind] -= 1
        counts[missing]  += 1

    # re-coerce fragment rules per room (answers changed)
    for rm in rooms:
        puzzles = []
        for rt in rm.get("routes") or []:
            q = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
            if q: puzzles.append(q)
        if puzzles:
            fr_rule = rm.get("fragment_rule") or "FIRST2"
            fr_rule, _ = _coerce_same_fragment_or_const_all(fr_rule, puzzles)
            rm["fragment_rule"] = fr_rule

    blob.setdefault("trail", {})["rooms"] = rooms
    return blob

def compose_trailroom(date_key: str, server_secret: str) -> Dict[str,Any]:
    salt = os.getenv("ESCAPE_REGEN_SALT", "")
    rng_seed = daily_seed(date_key, server_secret + salt)
    rng = rng_from_seed(rng_seed)
    if os.getenv("ESCAPE_MODEL","").lower()=="off" or os.getenv("ESCAPE_FORCE_OFFLINE","").lower() in ("1","true","yes"):
        return generate_room_offline(date_key, server_secret)

    # Do NOT reseed here; keep the salted RNG for the whole compose step
    blob = llm_generate_trailroom(date_key)
    if not isinstance(blob, dict):
        current_app.logger.warning("[escape] compose: LLM returned no blob; using offline for %s (model=%s)", date_key, os.getenv("ESCAPE_MODEL"))
        return generate_room_offline(date_key, server_secret)

    if "trail" not in blob and "rooms" in blob:
        blob = {
            "id": blob.get("id", date_key),
            "title": blob.get("title",""),
            "intro": blob.get("intro",""),
            "npc_lines": blob.get("npc_lines", []),
            "difficulty": blob.get("difficulty", DEFAULT_DIFFICULTY),
            "trail": {"supplies_start": blob.get("supplies_start", SUPPLIES_START_DEFAULT),
                      "rooms": blob.get("rooms", [])},
            "final": blob.get("final", {}),
            "anti_spoiler": blob.get("anti_spoiler", {"paraphrase_variants":3,"decoys":2}),
        }

    # 🔧 NEW: sanitize to 3 rooms / 3 routes per room
    blob = _sanitize_llm_blob(blob, rng, date_key)
    blob = _replace_recent_answers(blob, rng)  # <-- add this

    # Build a blacklist (for regen) from recent answers + stock
    blacklist = set(COMMON_STOCK_ANSWERS)
    try:
        for r in recent_rooms(ANSWER_COOLDOWN_DAYS):
            for rm in (r.json_blob or {}).get("trail", {}).get("rooms", []) or []:
                for rt in (rm.get("routes") or []):
                    sol = ((rt.get("puzzle") or {}).get("solution") or {}).get("answer")
                    if sol:
                        blacklist.add(normalize_answer(sol).lower())
            for p in (r.json_blob or {}).get("puzzles") or []:
                sol = (p.get("solution") or {}).get("answer")
                if sol:
                    blacklist.add(normalize_answer(sol).lower())
    except Exception:
        pass

    # Sanitize/repair LLM puzzles in-place (or regenerate deterministically)
    _sanitize_trail_puzzles(blob, rng, blacklist)

    # 🔁 NEW: reduce “same route → same mechanic” across scenes
    try:
        blob = _reshuffle_mechanics_for_variety(blob)
    except Exception as e:
        current_app.logger.warning("[escape] variety reshuffle skipped: %s", e)

    # NEW: last-mile fix so minis can't fail on pattern/answer mismatches
    try:
        blob = _fixup_minigames(blob, rng)
    except Exception as e:
        current_app.logger.warning("[escape] mini fixup skipped: %s", e)

    # ⭐ NEW: guarantee daily variety across all 9 minis (cover all 6 flavors)
    try:
        blob = _enforce_daily_mini_variety(blob, rng)
        # run the fixup one more time in case we swapped puzzles
        blob = _fixup_minigames(blob, rng)
    except Exception as e:
        current_app.logger.warning("[escape] enforce variety skipped: %s", e)

    # Validate & standardize; if broken, fallback offline
    try:
        # -- pre-validation hardening: purge trivial MCs and re-reroll recent answers
        try:
            for _rm in (blob.get("trail") or {}).get("rooms", []) or []:
                _theme = _rm.get("title") or _rm.get("text") or ""
                for _rt in (_rm.get("routes") or []):
                    _p = _rt.get("puzzle") if isinstance(_rt.get("puzzle"), dict) else {}
                    if (_p.get("type") == "mini") and (_p.get("mechanic") == "multiple_choice") and _looks_trivial_multiple_choice(_p):
                        _pid = (_p.get("id") or "mini_autogen")
                        _rt["puzzle"] = gen_pathcode(rng, _pid, set(), _theme).to_json()
            # run an extra cooldown-swap + fixup pass to avoid 'recently used' rejections
            blob = _replace_recent_answers(blob, rng)
            blob = _fixup_minigames(blob, rng)
        except Exception as _e:
            try: current_app.logger.warning("[escape] pre-validate hardening skipped: %s", _e)
            except Exception: pass
        room = validate_trailroom(blob)
    

        # Debug: show per-room fragments and final
        try:
            frags = []
            for rm in room.get("trail", {}).get("rooms", []):
                rule = rm.get("fragment_rule")
                p0 = (rm.get("routes", [{}])[0].get("puzzle") or {})
                ans0 = (p0.get("solution") or {}).get("answer", "")
                frags.append(apply_fragment_rule(ans0, rule))
            current_app.logger.info("[escape] fragments=%s final=%s",
                                    "-".join(frags),
                                    (room.get("final", {}).get("solution", {}) or {}).get("answer"))
        except Exception:
            pass

    except Exception as e:
        try: current_app.logger.error(f"[escape] validate_trailroom failed: {e}")

        except Exception: pass
        room = generate_room_offline(date_key, server_secret)

    if too_easy(room):
        room = harden(room, rng)

    try:
        if is_too_similar_to_recent(room, RECENT_WINDOW_DAYS):
            t2,i2 = rng.choice(THEMES); room["title"]=t2; room["intro"]=i2
    except Exception:
        pass

    try:
        room2 = llm_critic_patch(room)
        room = validate_trailroom(room2)  # keep trail validator
    except Exception:
        pass
    return room

def generate_room(date_key: str, server_secret: str) -> Dict[str, Any]:
    return compose_trailroom(date_key, server_secret)

# ───────────────────────── Public API (unchanged) ─────────────────────────

def ensure_daily_room(date_key: Optional[str] = None, force_regen: bool = False) -> Any:
    db, EscapeRoom = _get_db_and_models()
    if date_key is None: date_key = get_today_key()
    existing = db.session.query(EscapeRoom).filter_by(date_key=date_key).first()
    if existing and not force_regen:
        return existing

    secret = os.getenv("ESCAPE_SERVER_SECRET","dev_secret_change_me")
    for attempt in range(1, MAX_GEN_ATTEMPTS+1):
        try:
            room_blob = generate_room(date_key, secret)
            if existing:
                existing.json_blob = room_blob
                existing.difficulty = room_blob.get("difficulty", DEFAULT_DIFFICULTY)
                db.session.add(existing); db.session.commit(); return existing
            else:
                new_room = EscapeRoom(date_key=date_key, json_blob=room_blob,
                                      difficulty=room_blob.get("difficulty", DEFAULT_DIFFICULTY))
                db.session.add(new_room); db.session.commit(); return new_room
        except Exception as e:
            try: current_app.logger.warning(f"[escape] generation attempt {attempt} failed: {e}")
            except Exception: pass

    # last resort
    room_blob = generate_room_offline(date_key, secret)
    if existing:
        existing.json_blob = room_blob
        existing.difficulty = room_blob.get("difficulty", DEFAULT_DIFFICULTY)
        db.session.add(existing); db.session.commit(); return existing
    else:
        new_room = EscapeRoom(date_key=date_key, json_blob=room_blob,
                              difficulty=room_blob.get("difficulty", DEFAULT_DIFFICULTY))
        db.session.add(new_room); db.session.commit(); return new_room

def _match_answer(expected: str, submitted: str, pattern: Optional[str]) -> bool:
    if submitted is None: return False
    s = str(submitted).strip(); e = str(expected)
    if pattern and re.fullmatch(r"^\^\d+\$$", pattern):  # literal numeric lock
        return s == e
    return normalize_answer(s) == normalize_answer(e)

def _iter_all_puzzles(room_json: Dict[str,Any]):
    for rm in _as_dict_list(room_json.get("trail", {}).get("rooms")):
        for rt in _as_dict_list(rm.get("routes")):
            p = rt.get("puzzle") if isinstance(rt.get("puzzle"), dict) else {}
            if p: yield p
    for p in _as_dict_list(room_json.get("puzzles")):
        yield p

def _numeric_lock_accept_alt(answer: str, puzzle: Dict[str, Any]) -> bool:
    """
    If this is a numeric_lock and the prompt encodes constraints like:
      - the first two digits sum to S
      - the third digit is the first digit plus K1
      - the last/fourth digit equals the second digit plus K2
    then accept any 4-digit submission that satisfies them (e.g., both 0285 and 1194).
    """
    if not isinstance(puzzle, dict):
        return False
    if (puzzle.get("type") or puzzle.get("archetype")) != "numeric_lock":
        return False

    prompt = (puzzle.get("prompt") or "")
    # tolerant, case-insensitive parsing
    mS  = re.search(r"first\s+two\s+digits\s+sum\s+to\s+(-?\d+)", prompt, re.I)
    mK1 = re.search(r"third\s+digit\s+is\s+the\s+first\s+digit\s+plus\s+(-?\d+)", prompt, re.I)
    mK2 = re.search(r"(?:last|fourth)\s+digit\s+equals\s+the\s+second\s+digit\s+plus\s+(-?\d+)", prompt, re.I)
    if not (mS and mK1 and mK2):
        return False

    try:
        S  = int(mS.group(1))
        K1 = int(mK1.group(1))
        K2 = int(mK2.group(1))
    except Exception:
        return False

    digits = re.sub(r"\D", "", str(answer or ""))
    if len(digits) != 4:
        return False
    a, b, c, d = [int(ch) for ch in digits]
    return (a + b == S) and (c - a == K1) and (d - b == K2)

def verify_puzzle(room_json: Dict[str, Any], puzzle_id: str, answer: str) -> bool:
    for p in _iter_all_puzzles(room_json):
        if p.get("id") == puzzle_id:
            sol = (p.get("solution") or {}).get("answer")
            pattern = (p.get("answer_format") or {}).get("pattern")

            if sol is None:
                return False

            # Primary: exact stored answer (handles normalized/leading zeros)
            if _match_answer(str(sol), str(answer or ""), pattern):
                return True

            # Numeric lock tolerance: accept any valid 4-digit code that satisfies the prompt constraints.
            if _numeric_lock_accept_alt(answer, p):
                return True

            return False
    return False

def verify_meta_final(room_json: Dict[str, Any], submitted: str) -> bool:
    final = (room_json.get("final") or {}).get("solution", {})
    expect = final.get("answer") or room_json.get("final_code")
    if not expect: return False
    return normalize_answer(submitted) == normalize_answer(expect)

# ───────────────────────── Scheduler ─────────────────────────

_scheduler_started = False
def schedule_daily_generation(app) -> None:
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
    app.logger.info("[escape] scheduler started (daily 00:05 local).")
