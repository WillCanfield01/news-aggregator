# app/escape/core.py
# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Core Engine (OpenAI-enabled, hardened)

- Deterministic daily puzzle generation (seeded RNG)
- OpenAI theming (title/intro/inventory/difficulty) with strict JSON, safe overlay
- Algorithmic puzzle archetypes (anagram, caesar, vigenere, numeric lock)
- Guardrails: novelty window, answer cooldown, validation, "too easy" hardening
- Robust graph normalization to prevent bad shapes from LLM
- Guaranteed offline fallback

Public API:
  ensure_daily_room(date_key=None, force_regen=False) -> EscapeRoom
  verify_puzzle(room_json, puzzle_id, answer) -> bool
  schedule_daily_generation(app) -> None
"""

from __future__ import annotations

import os
import hmac
import hashlib
import random
import string
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import datetime as dt

import pytz
from flask import current_app

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Avoid circular imports. We only import db/models inside helpers.
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_and_models():
    """Late import to avoid circulars at module import time."""
    from app.extensions import db  # SQLAlchemy() singleton
    from .models import EscapeRoom  # ORM model
    return db, EscapeRoom


# ---------- Configurable constants ----------

RECENT_WINDOW_DAYS = 60
ANSWER_COOLDOWN_DAYS = 14
MAX_GEN_ATTEMPTS = 3
DEFAULT_DIFFICULTY = "medium"
TIMEZONE = os.getenv("ESCAPE_TZ", "America/Boise")

COMMON_STOCK_ANSWERS = {
    "piano", "keyboard", "silence", "time", "shadow", "map", "echo", "fire", "ice",
    "darkness", "light", "egg", "door", "wind", "river"
}

ANAGRAM_WORDS = [
    "lantern", "saffron", "garnet", "amplify", "banquet", "topaz", "onyx",
    "galaxy", "harbor", "cobalt", "jasper", "velvet", "sepia", "orchid",
    "monsoon", "rift", "cipher", "lilac", "ember", "quartz", "krypton",
    "zephyr", "coral", "indigo", "scarlet"
]

THEMES = [
    ("The Clockmaker’s Loft", "Dusty gears tick as you wake beneath copper skylights."),
    ("Signal in the Sublevel", "A faint hum from heavy conduits thrums through the floor."),
    ("Archive of Ash", "Charred shelves lean, cradling sealed folios that smell of smoke."),
    ("Night at the Conservatory", "Moonlight fractures through greenhouse panes onto damp stone."),
    ("The Salt Vault", "Clink… clink… droplets echo in a chalk-white storehouse."),
    ("Ferry to Nowhere", "River fog swallows the terminal as lanterns pulse and fade."),
    ("Radio Silence", "A dead station blinks a lone cursor on a green-glass screen."),
]

FINAL_CODE_MIN_LEN = 4
FINAL_CODE_MAX_LEN = 12


# ---------- Utility: deterministic seed ----------

def daily_seed(date_key: str, secret: str) -> int:
    digest = hmac.new(secret.encode("utf-8"), date_key.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def rng_from_seed(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r


def get_today_key(tz: Optional[str] = None) -> str:
    tzname = tz or TIMEZONE
    now = dt.datetime.now(pytz.timezone(tzname))
    return now.date().isoformat()


# ---------- Shingling & similarity ----------

def _shingles(text: str, k: int = 5) -> set:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < k:
        return {t}
    return {t[i:i+k] for i in range(len(t) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def recent_rooms(window_days: int = RECENT_WINDOW_DAYS) -> List[Any]:
    db, EscapeRoom = _get_db_and_models()
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=window_days)
    return (
        db.session.query(EscapeRoom)
        .filter(EscapeRoom.created_at >= cutoff)
        .order_by(EscapeRoom.created_at.desc())
        .limit(120)
        .all()
    )


def is_too_similar_to_recent(room_json: Dict[str, Any],
                             window_days: int = RECENT_WINDOW_DAYS,
                             sim_threshold: float = 0.35) -> bool:
    text = f"{room_json.get('title','')} {room_json.get('intro','')}"
    for p in room_json.get("puzzles", []):
        text += " " + p.get("prompt", "")
        for pv in p.get("paraphrases", []):
            text += " " + pv
    S = _shingles(text, k=7)

    for r in recent_rooms(window_days):
        blob = r.json_blob or {}
        t2 = f"{blob.get('title','')} {blob.get('intro','')}"
        for pp in (blob.get("puzzles") or []):
            t2 += " " + pp.get("prompt", "")
            for pv in pp.get("paraphrases", []):
                t2 += " " + pv
        S2 = _shingles(t2, k=7)
        if jaccard(S, S2) >= sim_threshold:
            return True
    return False


def answer_recently_used(answer: str, cooldown_days: int = ANSWER_COOLDOWN_DAYS) -> bool:
    db, EscapeRoom = _get_db_and_models()
    answer_norm = normalize_answer(answer)
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=cooldown_days)
    rs = db.session.query(EscapeRoom).filter(EscapeRoom.created_at >= cutoff).all()
    for r in rs:
        blob = r.json_blob or {}
        for p in blob.get("puzzles", []):
            sol = (p.get("solution") or {}).get("answer")
            if sol and normalize_answer(sol) == answer_norm:
                return True
    return False


# ---------- Normalization ----------

def normalize_answer(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (s or "")).upper()


# ---------- Algorithmic archetypes ----------

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

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "archetype": self.archetype,
            "prompt": self.prompt,
            "answer_format": self.answer_format,
            "solution": self.solution,
            "hints": self.hints,
            "decoys": self.decoys,
            "paraphrases": self.paraphrases,
        }


def _random_word(rng: random.Random, blacklist: set) -> str:
    pool = [w for w in ANAGRAM_WORDS if w not in blacklist]
    return rng.choice(pool) if pool else rng.choice(ANAGRAM_WORDS)


def gen_anagram(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    word = _random_word(rng, blacklist)
    letters = list(word)
    while True:
        rng.shuffle(letters)
        if "".join(letters) != word:
            break
    scrambled = "".join(letters)
    decoys = []
    for _ in range(2):
        l2 = letters[:]
        i, j = rng.randrange(len(l2)), rng.randrange(len(l2))
        l2[i], l2[j] = l2[j], l2[i]
        decoys.append("".join(l2))
    prompt = f"The note shows a scrambled word: **{scrambled}**. Unscramble it."
    hints = ["Look for common consonant clusters.", "Think color, material, or object."]
    paraphrases = [
        f"Desk anagram: {scrambled}. Restore it.",
        f"Jumbled letters read {scrambled}. What word?",
        f"A scrap shows {scrambled}. Unscramble to proceed.",
    ]
    return Puzzle(
        id=pid,
        archetype="anagram",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^[A-Za-z]{3,12}$"},
        solution={"answer": word},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases,
    )


def gen_caesar(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    candidates = [w for w in ANAGRAM_WORDS if 5 <= len(w) <= 8 and w not in blacklist]
    answer = rng.choice(candidates) if candidates else _random_word(rng, blacklist)
    shift = rng.randrange(1, 25)
    alphabet = string.ascii_uppercase

    def enc_caesar(txt: str, k: int) -> str:
        out = []
        for ch in txt.upper():
            if ch in alphabet:
                out.append(alphabet[(alphabet.index(ch) + k) % 26])
            else:
                out.append(ch)
        return "".join(out)

    ciphertext = enc_caesar(answer, shift)
    prompt = (
        f"A faded label reads: **{ciphertext}**.\n"
        f"It appears to be a Caesar shift. Enter the decoded word (A–Z only)."
    )
    hints = ["The shift is not 13.", "Try small positive shifts first."]
    decoys = [enc_caesar(answer, (shift + d) % 26) for d in (1, 2)]
    paraphrases = [
        f"The inscription {ciphertext} looks shifted. Restore the plaintext.",
        f"A rotating cipher hides a word: {ciphertext}. Undo the shift.",
        f"Decode {ciphertext} with a Caesar to reveal the word.",
    ]
    return Puzzle(
        id=pid,
        archetype="caesar",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": answer, "shift": shift},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases,
    )


def gen_vigenere(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    key_pool = [w for w in ANAGRAM_WORDS if 5 <= len(w) <= 8 and w not in blacklist]
    key = rng.choice(key_pool) if key_pool else "VELVET"

    code_candidates = [w for w in ANAGRAM_WORDS if 4 <= len(w) <= 8 and w not in blacklist and w.lower() != key.lower()]
    code = rng.choice(code_candidates) if code_candidates else "EMBER"
    phrase_templates = [
        f"THE CODEWORD IS {code.upper()} HIDDEN BETWEEN LINES",
        f"SEEK THE TOKEN {code.upper()} WITHIN THE NOTE",
        f"{code.upper()} IS THE CLUE IN PLAIN SIGHT",
    ]
    plaintext = rng.choice(phrase_templates)
    alphabet = string.ascii_uppercase

    def enc_vigenere(pt: str, k: str) -> str:
        out = []
        k = re.sub(r"[^A-Za-z]", "", k).upper()
        ki = 0
        for ch in pt.upper():
            if ch in alphabet:
                shift = alphabet.index(k[ki % len(k)])
                out.append(alphabet[(alphabet.index(ch) + shift) % 26])
                ki += 1
            else:
                out.append(ch)
        return "".join(out)

    ciphertext = enc_vigenere(plaintext, key)
    prompt = (
        "You find a strip of paper with a ciphered line:\n"
        f"**{ciphertext}**\n"
        "It looks like a Vigenère cipher. Enter the hidden CODEWORD (letters only)."
    )
    hints = [f"The key is a material/gem word ({len(key)} letters).", "Focus on uppercase words in plaintext."]
    decoys = [key.upper(), re.sub(r"[^A-Za-z]", "", plaintext.replace(code.upper(), "TOKEN"))[:len(code)]]
    paraphrases = [
        "A note encodes a sentence with Vigenère; recover the codeword.",
        "Decrypt the strip to find the embedded word you must submit.",
        "The plaintext hides a specific word—submit that word exactly.",
    ]
    return Puzzle(
        id=pid,
        archetype="vigenere",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": code, "key": key, "ciphertext": ciphertext},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases,
    )


def gen_numeric_lock(rng: random.Random, pid: str) -> Puzzle:
    d1 = rng.randrange(0, 10)
    d2 = rng.randrange(0, 10)
    d3 = rng.randrange(0, 10)
    d4 = rng.randrange(0, 10)

    c1 = f"The first two digits sum to {d1 + d2}."
    c2 = f"The third digit is the first digit plus {d3 - d1}."
    c3 = f"The last digit equals the second digit plus {d4 - d2}."

    code = f"{d1}{d2}{d3}{d4}"
    prompt = (
        "A keypad blinks awaiting a 4-digit code.\n"
        f"- {c1}\n- {c2}\n- {c3}\n"
        "Enter the full 4-digit code."
    )
    hints = ["Write the constraints as equations.", "Solve for the first digits, then propagate."]
    decoys = []
    for delta in (1, -1):
        nd1 = (d1 + delta) % 10
        decoys.append(f"{nd1}{d2}{d3}{d4}")
    paraphrases = [
        "The keypad expects four digits. Use the constraints to solve.",
        "Three numeric clues define the exact code—compute and enter it.",
        "A logic lock guards the door; derive the 4-digit answer.",
    ]
    return Puzzle(
        id=pid,
        archetype="numeric_lock",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^\d{4}$"},
        solution={"answer": code},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases,
    )


# ---------- OpenAI integration (title/intro theming) ----------

def _get_openai_client():
    # Allow forcing offline
    if os.getenv("ESCAPE_MODEL", "").lower() == "off" or os.getenv("ESCAPE_FORCE_OFFLINE", "").lower() in ("1", "true", "yes"):
        return None, None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        return client, "modern"
    except Exception:
        pass
    try:
        import openai  # type: ignore
        openai.api_key = api_key
        return openai, "legacy"
    except Exception:
        return None, None


def _architect_prompt_for_theme(algo_pack: List[Puzzle], date_key: str) -> str:
    """
    Ask only for title/intro/inventory/difficulty so we never risk mismatched puzzles.
    """
    # Summarize archetypes for extra variety without leaking answers
    archetype_list = [{"id": p.id, "archetype": p.archetype} for p in algo_pack]
    return (
        "You design the *atmosphere* for a daily mini escape room.\n"
        "Return ONLY a single minified JSON object with keys:\n"
        '{ "title": string, "intro": string, "inventory": [ { "id": string, "desc": string } ] (optional), '
        '"difficulty": "easy"|"medium"|"hard" (optional) }\n'
        "Constraints:\n"
        "- Fresh, non-cliché phrasing (no riddles like piano/time/shadow/etc.).\n"
        "- Keep it concise (title <= 60 chars; intro 1–3 short sentences).\n"
        "- DO NOT include any solutions or puzzle data. We'll supply puzzles server-side.\n"
        f"- DATE_KEY: {date_key}\n"
        f"- PUZZLE_ARCHETYPES: {json.dumps(archetype_list, ensure_ascii=False)}"
    )


def _extract_json_block(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    # Remove code fences if present
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        return t[i:j+1]
    return None


def llm_fetch_theme(algo_pack: List[Puzzle], date_key: str) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM ONLY for theme (title/intro/inventory/difficulty). Return None on failure.
    """
    client, mode = _get_openai_client()
    if not client:
        return None

    sys_msg = (
        "You are a concise puzzle theming assistant. "
        "Return STRICT JSON only—no markdown, no commentary."
    )
    user_content = _architect_prompt_for_theme(algo_pack, date_key)
    model = os.getenv("ESCAPE_MODEL", "gpt-4o-mini")

    try:
        text = ""
        if mode == "modern":
            # Prefer chat.completions with response_format (if supported)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.8,
                    response_format={"type": "json_object"},
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception:
                # Fallback to Responses API
                resp = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": sys_msg + "\n\n" + user_content}],
                    temperature=0.8,
                )
                text = (getattr(resp, "output_text", "") or "").strip()
        else:
            # Legacy
            text = client.ChatCompletion.create(  # type: ignore
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.8,
            )["choices"][0]["message"]["content"].strip()

        json_block = _extract_json_block(text)
        if not json_block:
            return None
        blob = json.loads(json_block)

        # minimal sanity
        if not isinstance(blob, dict):
            return None
        if "title" not in blob or "intro" not in blob:
            return None
        return blob
    except Exception as e:
        try:
            current_app.logger.warning(f"[escape] LLM theme fetch failed: {e}")
        except Exception:
            pass
        return None


def llm_critic_patch(room_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional critic: expects a JSON Patch (list of {op,path,value}). If anything goes wrong, return original.
    """
    client, mode = _get_openai_client()
    if not client:
        return room_json

    content = (
        "You are a puzzle critic. Inspect the JSON for clichés and very trivial solves. "
        "If minor edits can improve novelty without harming fairness, output a minimal JSON Patch "
        "array of operations with keys 'op' (replace/add), 'path' (JSON Pointer), and 'value'. "
        "If no changes are needed, return [].\n\n"
        f"ROOM_JSON:\n{json.dumps(room_json, ensure_ascii=False)}"
    )
    patch_ops: List[Dict[str, Any]] = []

    try:
        model = os.getenv("ESCAPE_MODEL", "gpt-4o-mini")
        text = ""
        if mode == "modern":
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or "[]"
            except Exception:
                resp = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": content}],
                    temperature=0.0,
                )
                text = getattr(resp, "output_text", "[]")
        else:
            text = client.ChatCompletion.create(  # type: ignore
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
            )["choices"][0]["message"]["content"]

        # Allow either a JSON array or a JSON object with "patch": [...]
        jb = _extract_json_block(text) or text
        parsed = json.loads(jb)
        if isinstance(parsed, dict) and "patch" in parsed:
            patch_ops = parsed["patch"]
        elif isinstance(parsed, list):
            patch_ops = parsed
        else:
            patch_ops = []
    except Exception as e:
        try:
            current_app.logger.info(f"[escape] LLM critic skipped: {e}")
        except Exception:
            pass
        return room_json

    try:
        blob = json.loads(json.dumps(room_json))  # deep copy

        # ⛔ Do not allow patches that touch core structures
        def _allowed(parts: List[str]) -> bool:
            if not parts:
                return False
            if parts[0] in {"puzzles", "graph", "final_code"}:
                return False
            return True

        for op in patch_ops:
            path = op.get("path", "")
            parts = [p for p in path.split("/") if p]
            if not _allowed(parts):
                continue
            if op.get("op") in ("replace", "add"):
                _json_path_set(blob, parts, op.get("value"))
        return blob
    except Exception:
        return room_json



def _json_path_set(blob: Dict[str, Any], parts: List[str], value: Any):
    cur = blob
    for i, p in enumerate(parts):
        is_last = i == len(parts) - 1
        if is_last:
            cur[p] = value
            return
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]


# ---------- Offline story wrapper (no LLM) ----------

def offline_wrap(algo_pack: List[Puzzle], date_key: str, rng: random.Random) -> Dict[str, Any]:
    title, intro = rng.choice(THEMES)
    node_ids = [p.id for p in algo_pack]
    nodes = [{"id": nid, "type": "puzzle", "requires": []} for nid in node_ids]
    final_gate = "final_door"
    nodes.append({"id": final_gate, "type": "gate", "requires": node_ids})

    return {
        "id": date_key,
        "title": title,
        "intro": intro,
        "graph": {"nodes": nodes, "start": ["room_intro"], "end": final_gate},
        "puzzles": [p.to_json() for p in algo_pack],
        "inventory": [{"id": "brass_key", "desc": "Worn and warm to the touch."}],
        "locks": [],
        "anti_spoiler": {"paraphrase_variants": 3, "decoys": 2},
        "difficulty": DEFAULT_DIFFICULTY,
    }


# ---------- Validation & hardening ----------

def validate_room(room_json: Dict[str, Any]) -> Dict[str, Any]:
    required_top = {"id", "title", "intro", "graph", "puzzles", "anti_spoiler", "difficulty"}
    missing = [k for k in required_top if k not in room_json]
    if missing:
        raise ValueError(f"Room missing keys: {missing}")

    if not isinstance(room_json["puzzles"], list) or not room_json["puzzles"]:
        raise ValueError("Room must contain at least one puzzle")

    seen_ids = set()
    for p in room_json["puzzles"]:
        for key in ("id", "archetype", "prompt", "answer_format", "solution"):
            if key not in p:
                raise ValueError(f"Puzzle missing '{key}'")
        pid = p["id"]
        if pid in seen_ids:
            raise ValueError("Duplicate puzzle id")
        seen_ids.add(pid)

        sol = p.get("solution", {})
        ans = sol.get("answer")
        if not ans or len(normalize_answer(ans)) == 0:
            raise ValueError("Puzzle has empty solution answer")

        if normalize_answer(ans).lower() in {s.upper() for s in COMMON_STOCK_ANSWERS}:
            raise ValueError("Puzzle uses stock/cliché answer")

        if answer_recently_used(str(ans)):
            raise ValueError("Answer recently used in the cooldown window")

        pattern = p.get("answer_format", {}).get("pattern")
        if pattern:
            if not re.match(r"^\^.*\$$", pattern):
                raise ValueError("answer_format.pattern must be anchored ^...$")
            if not re.match(pattern, str(ans)):
                if not re.match(pattern, normalize_answer(str(ans))):
                    raise ValueError("Solution does not match its declared pattern")

    graph = room_json.get("graph", {})
    if "end" not in graph:
        raise ValueError("Graph must define an 'end' gate id")

    fc = room_json.get("final_code")
    if fc:
        if not (FINAL_CODE_MIN_LEN <= len(normalize_answer(fc)) <= FINAL_CODE_MAX_LEN):
            raise ValueError("final_code length out of bounds")

    return room_json


def _normalize_graph(room: Dict[str, Any], puzzle_ids: List[str]) -> None:
    """
    Normalize graph shapes to avoid set() crashes and ensure end gate requires all puzzles.
    """
    g = room.setdefault("graph", {})
    nodes = g.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
        g["nodes"] = nodes

    seen: set = set()
    for i, n in enumerate(nodes):
        nid = n.get("id")
        if isinstance(nid, (str, int)):
            nid = str(nid)
        else:
            nid = f"auto_{i}"
        if nid in seen:
            nid = f"{nid}_{i}"
        n["id"] = nid
        seen.add(nid)
        req = n.get("requires", [])
        if not isinstance(req, list):
            req = []
        n["requires"] = [str(r) for r in req]
        n.setdefault("type", "puzzle")

    end_id = g.get("end")
    if not isinstance(end_id, (str, int)):
        end_id = "final_door"
        g["end"] = end_id
    end_id = str(end_id)

    end_node = None
    for n in nodes:
        if n.get("id") == end_id:
            end_node = n
            break
    if end_node is None:
        end_node = {"id": end_id, "type": "gate", "requires": list(puzzle_ids)}
        nodes.append(end_node)
    else:
        end_node["type"] = end_node.get("type", "gate")
        end_node["requires"] = list(puzzle_ids)


def too_easy(room_json: Dict[str, Any]) -> bool:
    puzzles = room_json.get("puzzles", [])
    if len(puzzles) <= 1:
        return True
    easy_count = 0
    for p in puzzles:
        arch = p.get("archetype")
        ans = normalize_answer(p.get("solution", {}).get("answer", ""))
        pattern = (p.get("answer_format") or {}).get("pattern", "")
        if arch in {"anagram", "caesar"} and 4 <= len(ans) <= 6:
            easy_count += 1
        if arch == "numeric_lock" and pattern == r"^\d{4}$":
            easy_count += 1
        if ans in {"1234", "1111", "0000", "2580"}:
            easy_count += 1
    return easy_count >= len(puzzles)


def harden(room_json: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    if rng is None:
        rng = random.Random()
    for p in room_json.get("puzzles", []):
        decs = p.get("decoys") or []
        while len(decs) < 3:
            ans = str(p.get("solution", {}).get("answer", ""))
            if p.get("archetype") in {"anagram", "caesar", "vigenere"} and ans.isalpha():
                letters = list(ans)
                rng.shuffle(letters)
                decs.append("".join(letters))
            elif re.match(r"^\d+$", ans):
                num = int(ans)
                tweak = (num + rng.randrange(1, 9)) % (10 ** len(ans))
                decs.append(str(tweak).zfill(len(ans)))
            else:
                decs.append(ans[::-1])
        p["decoys"] = decs

    if not room_json.get("final_code"):
        pieces: List[str] = []
        for p in room_json.get("puzzles", []):
            ans = str(p.get("solution", {}).get("answer", ""))
            if re.match(r"^\d+$", ans):
                pieces.append(ans[-2:])
            else:
                pieces.append(normalize_answer(ans)[:2])
        meta = "".join(pieces)
        meta = meta[:FINAL_CODE_MAX_LEN]
        if len(meta) < FINAL_CODE_MIN_LEN:
            meta = (meta + "X" * FINAL_CODE_MIN_LEN)[:FINAL_CODE_MAX_LEN]
        room_json["final_code"] = meta
        end_id = room_json.get("graph", {}).get("end", "final_door")
        room_json["graph"]["end"] = end_id
        room_json.setdefault("meta_gate", {"requires_all": True, "expects": "final_code"})

    room_json["difficulty"] = "hard"
    anti = room_json.get("anti_spoiler", {})
    anti["decoys"] = max(3, anti.get("decoys", 2))
    room_json["anti_spoiler"] = anti
    return room_json


# ---------- Room assembly pipeline ----------

def build_archetype_mix(rng: random.Random) -> List[str]:
    pool = ["anagram", "caesar", "vigenere", "numeric_lock"]
    rng.shuffle(pool)
    return pool[:3]


def build_algorithmic_pack(rng: random.Random, blacklist: set) -> List[Puzzle]:
    mix = build_archetype_mix(rng)
    pack: List[Puzzle] = []
    for idx, archetype in enumerate(mix, start=1):
        pid = f"pz_{idx}"
        if archetype == "anagram":
            pack.append(gen_anagram(rng, pid, blacklist))
        elif archetype == "caesar":
            pack.append(gen_caesar(rng, pid, blacklist))
        elif archetype == "vigenere":
            pack.append(gen_vigenere(rng, pid, blacklist))
        elif archetype == "numeric_lock":
            pack.append(gen_numeric_lock(rng, pid))
    if not pack:
        pack = [gen_numeric_lock(rng, "pz_1")]
    return pack


def compose_room(algo_pack: List[Puzzle], date_key: str, rng: random.Random) -> Dict[str, Any]:
    """
    Base offline structure + (optional) LLM theming overlay.
    We keep puzzles/graph server-authored for correctness and anti-cheat.
    """
    room = offline_wrap(algo_pack, date_key, rng)
    theme = llm_fetch_theme(algo_pack, date_key)
    if isinstance(theme, dict):
        # Overlay approved fields only
        if isinstance(theme.get("title"), str) and theme["title"].strip():
            room["title"] = theme["title"].strip()[:60]
        if isinstance(theme.get("intro"), str) and theme["intro"].strip():
            room["intro"] = re.sub(r"\s+", " ", theme["intro"].strip())
        if isinstance(theme.get("inventory"), list):
            room["inventory"] = theme["inventory"][:5]  # keep it short
        if theme.get("difficulty") in ("easy", "medium", "hard"):
            room["difficulty"] = theme["difficulty"]
    return room


def generate_room_offline(date_key: str, server_secret: str) -> Dict[str, Any]:
    seed = daily_seed(date_key, server_secret)
    rng = rng_from_seed(seed)

    blacklist = set(COMMON_STOCK_ANSWERS)
    try:
        for r in recent_rooms(ANSWER_COOLDOWN_DAYS):
            for p in (r.json_blob or {}).get("puzzles", []):
                sol = (p.get("solution") or {}).get("answer")
                if sol:
                    blacklist.add(normalize_answer(sol).lower())
    except Exception:
        pass

    pack = build_algorithmic_pack(rng, blacklist)
    room = offline_wrap(pack, date_key, rng)

    puzzle_ids = [str(p.get("id")) for p in room.get("puzzles", []) if p.get("id")]
    _normalize_graph(room, puzzle_ids)

    room = validate_room(room)
    if too_easy(room):
        room = harden(room, rng)
        room = validate_room(room)
        
    # Apply critic safely; revert on any validation failure
    pre_critic = json.loads(json.dumps(room))  # deep copy
    try:
        patched = llm_critic_patch(pre_critic)
        validate_room(patched)
        room = patched
    except Exception as e:
        try:
            current_app.logger.info(f"[escape] critic patch rejected; keeping pre-critic room: {e}")
        except Exception:
            pass
        room = pre_critic

    return room


def generate_room(date_key: str, server_secret: str) -> Dict[str, Any]:
    """
    Primary generator (OpenAI theming if enabled) with robust fallbacks.
    """
    # Force offline if requested
    if os.getenv("ESCAPE_MODEL", "").lower() == "off" or os.getenv("ESCAPE_FORCE_OFFLINE", "").lower() in ("1", "true", "yes"):
        return generate_room_offline(date_key, server_secret)

    seed = daily_seed(date_key, server_secret)
    rng = rng_from_seed(seed)

    # Build blacklist from recent answers
    blacklist = set(COMMON_STOCK_ANSWERS)
    for r in recent_rooms(ANSWER_COOLDOWN_DAYS):
        for p in (r.json_blob or {}).get("puzzles", []):
            sol = (p.get("solution") or {}).get("answer")
            if sol:
                blacklist.add(normalize_answer(sol).lower())

    pack = build_algorithmic_pack(rng, blacklist)

    # Base room + LLM overlay (safe fields only)
    room = compose_room(pack, date_key, rng)

    # Normalize graph before any set() calls
    puzzle_ids = [str(p.get("id")) for p in room.get("puzzles", []) if p.get("id")]
    _normalize_graph(room, puzzle_ids)

    # Validate; harden if needed
    try:
        room = validate_room(room)
    except Exception as e:
        try:
            current_app.logger.error(f"[escape] validate_room failed (primary): {e}; puzzles={len(room.get('puzzles') or [])}")
        except Exception:
            pass
        # Absolute fallback
        room = generate_room_offline(date_key, server_secret)

    if too_easy(room):
        room = harden(room, rng)
        room = validate_room(room)

    # Novelty tweak + (optional) critic
    try:
        if is_too_similar_to_recent(room, RECENT_WINDOW_DAYS):
            t2, i2 = rng.choice(THEMES)
            room["title"] = t2
            room["intro"] = i2
    except Exception:
        pass

    try:
        room = llm_critic_patch(room)
        room = validate_room(room)
    except Exception:
        pass

    return room


# ---------- Public API: ensure/caching ----------

def ensure_daily_room(date_key: Optional[str] = None, force_regen: bool = False) -> Any:
    db, EscapeRoom = _get_db_and_models()
    if date_key is None:
        date_key = get_today_key()

    existing = db.session.query(EscapeRoom).filter_by(date_key=date_key).first()
    if existing and not force_regen:
        return existing

    secret = os.getenv("ESCAPE_SERVER_SECRET", "dev_secret_change_me")
    last_error = None

    for attempt in range(1, MAX_GEN_ATTEMPTS + 1):
        try:
            room_blob = generate_room(date_key, secret)
            if existing:
                existing.json_blob = room_blob
                existing.difficulty = room_blob.get("difficulty", DEFAULT_DIFFICULTY)
                db.session.add(existing)
                db.session.commit()
                return existing
            else:
                new_room = EscapeRoom(
                    date_key=date_key,
                    json_blob=room_blob,
                    difficulty=room_blob.get("difficulty", DEFAULT_DIFFICULTY),
                )
                db.session.add(new_room)
                db.session.commit()
                return new_room
        except Exception as e:
            last_error = e
            try:
                current_app.logger.warning(f"[escape] generation attempt {attempt} failed: {e}")
            except Exception:
                pass

    # Absolute offline fallback
    room_blob = generate_room_offline(date_key, secret)
    if existing:
        existing.json_blob = room_blob
        existing.difficulty = room_blob.get("difficulty", DEFAULT_DIFFICULTY)
        db.session.add(existing)
        db.session.commit()
        return existing
    else:
        new_room = EscapeRoom(
            date_key=date_key,
            json_blob=room_blob,
            difficulty=room_blob.get("difficulty", DEFAULT_DIFFICULTY),
        )
        db.session.add(new_room)
        db.session.commit()
        return new_room


# ---------- Verify answers ----------

def _match_answer(expected: str, submitted: str, pattern: Optional[str]) -> bool:
    if submitted is None:
        return False
    s = submitted.strip()
    e = str(expected)
    if pattern and re.fullmatch(r"^\^\d+\$$", pattern):
        return s == e
    return normalize_answer(s) == normalize_answer(e)


def verify_puzzle(room_json: Dict[str, Any], puzzle_id: str, answer: str) -> bool:
    for p in room_json.get("puzzles", []):
        if p.get("id") == puzzle_id:
            sol = (p.get("solution") or {}).get("answer")
            pattern = (p.get("answer_format") or {}).get("pattern")
            if sol is None:
                return False
            return _match_answer(str(sol), str(answer or ""), pattern)
    return False


def verify_meta_final(room_json: Dict[str, Any], submitted: str) -> bool:
    expect = room_json.get("final_code")
    if not expect:
        return False
    return normalize_answer(submitted) == normalize_answer(expect)


# ---------- Scheduler hook ----------

_scheduler_started = False

def schedule_daily_generation(app) -> None:
    """Start APScheduler to generate the daily room just after local midnight."""
    global _scheduler_started
    if _scheduler_started:
        try:
            app.logger.info("[escape] scheduler already started; skipping.")
        except Exception:
            pass
        return
    _scheduler_started = True

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as e:
        try:
            app.logger.warning(f"[escape] APScheduler not available: {e}")
        except Exception:
            pass
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
