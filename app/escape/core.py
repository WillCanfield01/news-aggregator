# app/escape/core.py
# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Core Engine (MVP, condensed)

Responsibilities:
- Deterministic daily room generation via seed (HMAC(date, server_secret))
- Algorithmic puzzle archetypes (cipher, anagram, numeric lock)
- Optional LLM "story + clue" wrapper with architect/critic passes (fallback offline if no API)
- Similarity + answer-collision guardrails (last 60 days)
- Heuristic solver sanity check; "harden" if too easy
- JSON schema validation (lightweight, explicit checks)
- Exposed functions:
    - ensure_daily_room(date_key=None) -> EscapeRoom (creates/caches in DB)
    - verify_puzzle(room_json, puzzle_id, answer) -> bool
    - schedule_daily_generation(app)  # APScheduler cron at local midnight

You may tweak thresholds and archetype pools over time.
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
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt

import pytz
from flask import current_app

# DB + models
from app import db
from .models import EscapeRoom  # EscapeAttempt is used by routes, not required here


# ---------- Configurable constants ----------

RECENT_WINDOW_DAYS = 60
ANSWER_COOLDOWN_DAYS = 14
MAX_GEN_ATTEMPTS = 3
DEFAULT_DIFFICULTY = "medium"
TIMEZONE = os.getenv("ESCAPE_TZ", "America/Boise")

# Common riddle answers to avoid; these are flagged in validation/hardening
COMMON_STOCK_ANSWERS = {
    "piano", "keyboard", "silence", "time", "shadow", "map", "echo", "fire", "ice",
    "darkness", "light", "egg", "door", "wind", "river"
}

# Word pools for anagrams that avoid cliché answers (you can expand)
ANAGRAM_WORDS = [
    "lantern", "saffron", "garnet", "amplify", "banquet", "topaz", "onyx",
    "galaxy", "harbor", "cobalt", "jasper", "velvet", "sepia", "orchid",
    "monsoon", "rift", "cipher", "lilac", "ember", "quartz", "krypton",
    "zephyr", "coral", "indigo", "scarlet"
]

# Key names/items/themes for variety (used in offline story wrapper)
THEMES = [
    ("The Clockmaker’s Loft", "Dusty gears tick as you wake beneath copper skylights."),
    ("Signal in the Sublevel", "A faint hum from heavy conduits thrums through the floor."),
    ("Archive of Ash", "Charred shelves lean, cradling sealed folios that smell of smoke."),
    ("Night at the Conservatory", "Moonlight fractures through greenhouse panes onto damp stone."),
    ("The Salt Vault", "Clink… clink… droplets echo in a chalk-white storehouse."),
    ("Ferry to Nowhere", "River fog swallows the terminal as lanterns pulse and fade."),
    ("Radio Silence", "A dead station blinks a lone cursor on a green-glass screen."),
]

# Final answer formatting guardrails
FINAL_CODE_MIN_LEN = 4
FINAL_CODE_MAX_LEN = 12


# ---------- Utility: deterministic seed ----------

def daily_seed(date_key: str, secret: str) -> int:
    """
    Deterministically derive a 64-bit seed from date_key and a server secret.
    """
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


def recent_rooms(window_days: int = RECENT_WINDOW_DAYS) -> List[EscapeRoom]:
    """
    Pull recent rooms to compare novelty and answers.
    """
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
    """
    Compare title+intro and clue texts vs recent rooms via character shingles.
    """
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
    """
    Avoid identical answers repeating within a short window.
    """
    answer_norm = normalize_answer(answer)
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=cooldown_days)
    rs = (
        db.session.query(EscapeRoom)
        .filter(EscapeRoom.created_at >= cutoff)
        .all()
    )
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


# ---------- Algorithmic archetypes (deterministic by seed) ----------

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
    # ensure scramble isn't identical
    while True:
        rng.shuffle(letters)
        if "".join(letters) != word:
            break
    scrambled = "".join(letters)
    decoys = []
    # generate two plausible decoys by swapping pairs
    for _ in range(2):
        l2 = letters[:]
        i, j = rng.randrange(len(l2)), rng.randrange(len(l2))
        l2[i], l2[j] = l2[j], l2[i]
        decoys.append("".join(l2))

    prompt = f"The note shows a scrambled word: **{scrambled}**. Unscramble it to form a valid English word."
    hints = [f"Look for common consonant clusters.", f"It relates loosely to color/materials or objects."]
    paraphrases = [
        f"On the desk is an anagram: {scrambled}. Restore it.",
        f"You spot letters jumbled into {scrambled}. What is the word?",
        f"A scrap shows {scrambled}. Unscramble to proceed."
    ]
    return Puzzle(
        id=pid,
        archetype="anagram",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^[A-Za-z]{3,12}$"},
        solution={"answer": word},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases
    )


def gen_caesar(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    # choose a non-stock answer 5-8 letters
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
    hints = [f"The shift is not 13.", f"Try small positive shifts first."]
    decoys = [enc_caesar(answer, (shift + d) % 26) for d in (1, 2)]
    paraphrases = [
        f"The inscription {ciphertext} looks shifted. What plaintext restores it?",
        f"A rotating cipher hides a word: {ciphertext}. Undo the shift.",
        f"Decode {ciphertext} with a Caesar to reveal the word."
    ]
    return Puzzle(
        id=pid,
        archetype="caesar",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": answer, "shift": shift},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases
    )


def gen_vigenere(rng: random.Random, pid: str, blacklist: set) -> Puzzle:
    # choose a word to *extract* as final answer from plaintext; use key with length 5-7
    key_len = rng.randrange(5, 8)
    # pick a key distinct from answer candidates
    key_pool = [w for w in ANAGRAM_WORDS if 5 <= len(w) <= 8 and w not in blacklist]
    key = rng.choice(key_pool) if key_pool else "VELVET"

    # plaintext phrase: embed a 4-8 letter code (letters only)
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
    hints = [f"The key is a material or gem-like word ({len(key)} letters).", "Focus on uppercase words in the plaintext."]
    decoys = [key.upper(), re.sub(r"[^A-Za-z]", "", plaintext.replace(code.upper(), "TOKEN"))[:len(code)]]
    paraphrases = [
        "A note encodes a sentence with Vigenère; recover the codeword.",
        "Decrypt the strip to find the embedded word you must submit.",
        "The plaintext hides a specific word—submit that word exactly."
    ]
    return Puzzle(
        id=pid,
        archetype="vigenere",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": code, "key": key, "ciphertext": ciphertext},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases
    )


def gen_numeric_lock(rng: random.Random, pid: str) -> Puzzle:
    """
    Generate a solvable 4-digit code with constraints.
    """
    d1 = rng.randrange(0, 10)
    d2 = rng.randrange(0, 10)
    d3 = rng.randrange(0, 10)
    d4 = rng.randrange(0, 10)

    # Add constraints that are consistent (not necessarily unique, but good enough for MVP)
    c1 = f"The first two digits sum to {d1 + d2}."
    c2 = f"The third digit is the first digit plus {d3 - d1}."
    c3 = f"The last digit equals the second digit plus {d4 - d2}."

    code = f"{d1}{d2}{d3}{d4}"
    prompt = (
        "A keypad blinks awaiting a 4-digit code.\n"
        f"- {c1}\n- {c2}\n- {c3}\n"
        "Enter the full 4-digit code."
    )
    hints = ["Try writing the constraints as equations.", "Solve for the first digits, then propagate."]
    decoys = []
    # nearby decoys
    for delta in (1, -1):
        nd1 = (d1 + delta) % 10
        decoys.append(f"{nd1}{d2}{d3}{d4}")
    paraphrases = [
        "The keypad expects four digits. Use the constraints to solve.",
        "Three numeric clues define the exact code—compute and enter it.",
        "A logic lock guards the door; derive the 4-digit answer."
    ]
    return Puzzle(
        id=pid,
        archetype="numeric_lock",
        prompt=prompt,
        answer_format={"type": "string", "pattern": r"^\d{4}$"},
        solution={"answer": code},
        hints=hints,
        decoys=decoys,
        paraphrases=paraphrases
    )


# ---------- Optional LLM wrapper (architect + critic) ----------

def _get_openai_client():
    """
    Try to initialize an OpenAI client if available, otherwise return None.
    Supports both new 'openai.OpenAI()' and legacy 'openai' interface.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    # prefer modern client
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        return client, "modern"
    except Exception:
        pass

    # fallback legacy
    try:
        import openai  # type: ignore
        openai.api_key = api_key
        return openai, "legacy"
    except Exception:
        return None, None


def _architect_prompt(algo_pack: List[Puzzle], date_key: str) -> str:
    """
    Compose a system+user prompt that asks the LLM to wrap puzzles in story JSON.
    We'll feed the algorithmic puzzles (with answers) and ask it to output structured JSON
    conforming to our expected keys (title, intro, graph, puzzles, etc.).
    """
    base = (
        "You are a puzzle architect. You wrap algorithmic puzzles into a coherent, fair, "
        "novel daily escape room. Avoid stock riddles like 'piano', 'keyboard', 'silence', 'time', 'shadow', 'map', 'echo'. "
        "Vary answer formats; include 2 misleading but fair decoys per puzzle and 3 paraphrases of each prompt. "
        "Ensure a clear dependency DAG with exactly one final gate requiring all puzzles solved. "
        "Return STRICT JSON with keys: id, title, intro, graph:{nodes[],start[],end}, puzzles[], inventory[], locks[], "
        "anti_spoiler:{paraphrase_variants,decoys}, difficulty. Include puzzle.solution as provided."
    )
    # Pass algo details to the LLM
    algo_json = json.dumps([p.to_json() for p in algo_pack], ensure_ascii=False)
    return (
        f"{base}\n\n"
        f"DATE_KEY: {date_key}\n"
        f"ALGO_PUZZLES_JSON:\n{algo_json}\n\n"
        "Rules:\n"
        "- Keep language concise and atmospheric.\n"
        "- Do not reveal solutions in the title/intro.\n"
        "- Final gate unlocks only when all puzzle ids have been correctly answered.\n"
    )


def _critic_prompt(room_json: Dict[str, Any]) -> str:
    return (
        "You are a puzzle critic. Inspect the JSON for clichés, repeated patterns, "
        "and trivial solves. If minor edits can increase novelty/difficulty without compromising fairness, "
        "output a minimal JSON Patch as a list of operations with 'op', 'path', and 'value'. "
        "If no changes needed, output [].\n\n"
        f"ROOM_JSON:\n{json.dumps(room_json, ensure_ascii=False)}"
    )


def llm_wrap_story_and_clues(algo_pack: List[Puzzle], date_key: str) -> Optional[Dict[str, Any]]:
    client, mode = _get_openai_client()
    if not client:
        return None  # offline wrapper will be used

    content = _architect_prompt(algo_pack, date_key)
    try:
        if mode == "modern":
            # Try Responses API first
            try:
                resp = client.responses.create(
                    model=os.getenv("ESCAPE_MODEL", "gpt-4.1-mini"),
                    input=[{"role": "user", "content": content}],
                    temperature=0.8,
                )
                text = resp.output_text  # type: ignore
            except Exception:
                # Fallback to chat.completions if responses failed
                resp = client.chat.completions.create(
                    model=os.getenv("ESCAPE_MODEL", "gpt-4.1-mini"),
                    messages=[{"role": "user", "content": content}],
                    temperature=0.8,
                )
                text = resp.choices[0].message.content  # type: ignore
        else:
            # legacy openai
            text = client.ChatCompletion.create(  # type: ignore
                model=os.getenv("ESCAPE_MODEL", "gpt-4.1-mini"),
                messages=[{"role": "user", "content": content}],
                temperature=0.8,
            )["choices"][0]["message"]["content"]
        # Parse JSON block in response
        text = text.strip()
        # Try to extract JSON if the model wrapped it in formatting
        match = re.search(r"\{.*\}\s*$", text, re.S)
        blob = json.loads(match.group(0) if match else text)
        return blob
    except Exception as e:
        current_app.logger.warning(f"[escape] LLM architect failed: {e}")
        return None


def llm_critic_patch(room_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a lightweight critic; if LLM unavailable, return the original unmodified.
    We apply a JSON patch if provided correctly; else ignore.
    """
    client, mode = _get_openai_client()
    if not client:
        return room_json

    content = _critic_prompt(room_json)
    patch_ops: List[Dict[str, Any]] = []
    try:
        if mode == "modern":
            try:
                resp = client.responses.create(
                    model=os.getenv("ESCAPE_MODEL", "gpt-4.1-mini"),
                    input=[{"role": "user", "content": content}],
                    temperature=0.0,
                )
                text = resp.output_text  # type: ignore
            except Exception:
                resp = client.chat.completions.create(
                    model=os.getenv("ESCAPE_MODEL", "gpt-4.1-mini"),
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                )
                text = resp.choices[0].message.content  # type: ignore
        else:
            text = client.ChatCompletion.create(  # type: ignore
                model=os.getenv("ESCAPE_MODEL", "gpt-4.1-mini"),
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
            )["choices"][0]["message"]["content"]
        patch_ops = json.loads(text)
    except Exception as e:
        current_app.logger.info(f"[escape] LLM critic skipped: {e}")
        return room_json

    try:
        # Basic JSON Patch (limited to replace/add)
        blob = json.loads(json.dumps(room_json))  # deep copy
        for op in patch_ops:
            path = op.get("path", "")
            parts = [p for p in path.split("/") if p]
            if op.get("op") == "replace":
                _json_path_set(blob, parts, op.get("value"))
            elif op.get("op") == "add":
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


# ---------- Offline story wrapper (if no LLM) ----------

def offline_wrap(algo_pack: List[Puzzle], date_key: str, rng: random.Random) -> Dict[str, Any]:
    """
    Produce a coherent room JSON without any external model.
    We paraphrase and theme from fixed pools.
    """
    title, intro = rng.choice(THEMES)
    # Build dependency: all puzzles must be solved, then final gate opens
    node_ids = [p.id for p in algo_pack]
    nodes = [{"id": nid, "type": "puzzle", "requires": []} for nid in node_ids]
    final_gate = "final_door"
    nodes.append({"id": final_gate, "type": "gate", "requires": node_ids})

    room = {
        "id": date_key,
        "title": title,
        "intro": intro,
        "graph": {"nodes": nodes, "start": ["room_intro"], "end": final_gate},
        "puzzles": [p.to_json() for p in algo_pack],
        "inventory": [
            {"id": "brass_key", "desc": "Worn and warm to the touch."},
        ],
        "locks": [],
        "anti_spoiler": {"paraphrase_variants": 3, "decoys": 2},
        "difficulty": DEFAULT_DIFFICULTY,
    }
    return room


# ---------- Validation ----------

def validate_room(room_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal explicit validation; raises ValueError if unacceptable.
    """
    required_top = {"id", "title", "intro", "graph", "puzzles", "anti_spoiler", "difficulty"}
    missing = [k for k in required_top if k not in room_json]
    if missing:
        raise ValueError(f"Room missing keys: {missing}")

    if not isinstance(room_json["puzzles"], list) or not room_json["puzzles"]:
        raise ValueError("Room must contain at least one puzzle")

    # Validate puzzle fields & answer cooldowns
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

        # Avoid stock clichés
        if normalize_answer(ans).lower() in {s.upper() for s in COMMON_STOCK_ANSWERS}:
            raise ValueError("Puzzle uses stock/cliché answer")

        # Apply answer cooldown
        if answer_recently_used(str(ans)):
            raise ValueError("Answer recently used in the cooldown window")

        # Pattern sanity
        pattern = p.get("answer_format", {}).get("pattern")
        if pattern:
            if not re.match(r"^\^.*\$$", pattern):
                raise ValueError("answer_format.pattern must be anchored ^...$")
            # Quick check: the solution should match its own pattern
            if not re.match(pattern, str(ans)):
                # Permit normalization mismatch (e.g., plaintext vs uppercase)
                if not re.match(pattern, normalize_answer(str(ans))):
                    raise ValueError("Solution does not match its declared pattern")

    # Graph end gate required
    graph = room_json.get("graph", {})
    if "end" not in graph:
        raise ValueError("Graph must define an 'end' gate id")

    # Final code sanity (if present)
    fc = room_json.get("final_code")
    if fc:
        if not (FINAL_CODE_MIN_LEN <= len(normalize_answer(fc)) <= FINAL_CODE_MAX_LEN):
            raise ValueError("final_code length out of bounds")

    return room_json


# ---------- Heuristic solver & hardening ----------

def too_easy(room_json: Dict[str, Any]) -> bool:
    """
    Heuristic: flag trivially easy rooms (e.g., all puzzles single-step with common patterns).
    """
    puzzles = room_json.get("puzzles", [])
    if len(puzzles) <= 1:
        return True  # a single puzzle is too easy for "daily" feel

    easy_count = 0
    for p in puzzles:
        arch = p.get("archetype")
        ans = normalize_answer(p.get("solution", {}).get("answer", ""))
        pattern = (p.get("answer_format") or {}).get("pattern", "")

        if arch in {"anagram", "caesar"} and 4 <= len(ans) <= 6:
            easy_count += 1
        if arch == "numeric_lock" and pattern == r"^\d{4}$":
            easy_count += 1

        # common sequences
        if ans in {"1234", "1111", "0000", "2580"}:
            easy_count += 1

    return easy_count >= len(puzzles)  # if all are "easy-ish", mark too easy


def harden(room_json: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """
    Increase difficulty slightly: add more decoys, extend hints delay (client can use),
    and optionally add a light-weight meta step (final code = concat of first letters/digits).
    """
    if rng is None:
        rng = random.Random()

    # Add decoys to all puzzles
    for p in room_json.get("puzzles", []):
        decs = p.get("decoys") or []
        while len(decs) < 3:
            # fabricate a decoy by shuffling or tweaking answer
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

    # Add a simple meta "final_code" if not present:
    if not room_json.get("final_code"):
        pieces: List[str] = []
        for p in room_json.get("puzzles", []):
            ans = str(p.get("solution", {}).get("answer", ""))
            if re.match(r"^\d+$", ans):
                pieces.append(ans[-2:])  # last two digits
            else:
                pieces.append(normalize_answer(ans)[:2])  # first two letters
        meta = "".join(pieces)
        # clip within bounds
        meta = meta[:FINAL_CODE_MAX_LEN]
        if len(meta) < FINAL_CODE_MIN_LEN:
            meta = (meta + "X" * FINAL_CODE_MIN_LEN)[:FINAL_CODE_MAX_LEN]
        room_json["final_code"] = meta
        # Wrap a meta gate in the graph (client should prompt after all puzzles)
        end_id = room_json.get("graph", {}).get("end", "final_door")
        room_json["graph"]["end"] = end_id  # ensure exists
        room_json.setdefault("meta_gate", {"requires_all": True, "expects": "final_code"})

    # Increase conceptual difficulty
    room_json["difficulty"] = "hard"
    # Anti-spoiler: increment decoys target
    anti = room_json.get("anti_spoiler", {})
    anti["decoys"] = max(3, anti.get("decoys", 2))
    room_json["anti_spoiler"] = anti
    return room_json


# ---------- Room assembly pipeline ----------

def build_archetype_mix(rng: random.Random) -> List[str]:
    """
    Rotate across a small pool; guarantee diversity.
    """
    pool = ["anagram", "caesar", "vigenere", "numeric_lock"]
    rng.shuffle(pool)
    # pick 3 distinct types per day
    return pool[:3]


def build_algorithmic_pack(rng: random.Random, blacklist: set) -> List[Puzzle]:
    """
    Create a set of puzzle objects (answers known).
    """
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
    return pack


def compose_room(algo_pack: List[Puzzle], date_key: str, rng: random.Random) -> Dict[str, Any]:
    """
    Try LLM wrapper; fallback to offline wrapper.
    """
    blob = llm_wrap_story_and_clues(algo_pack, date_key)
    if blob is None:
        blob = offline_wrap(algo_pack, date_key, rng)
    return blob


def generate_room(date_key: str, server_secret: str) -> Dict[str, Any]:
    """
    Full generation pipeline with guardrails and hardening.
    """
    seed = daily_seed(date_key, server_secret)
    rng = rng_from_seed(seed)

    # Populate blacklist from recent answers to pre-empt cooldown failures
    blacklist = set(COMMON_STOCK_ANSWERS)
    for r in recent_rooms(ANSWER_COOLDOWN_DAYS):
        for p in (r.json_blob or {}).get("puzzles", []):
            sol = (p.get("solution") or {}).get("answer")
            if sol:
                blacklist.add(normalize_answer(sol).lower())

    pack = build_algorithmic_pack(rng, blacklist)
    room = compose_room(pack, date_key, rng)

    # Ensure graph "end" requires all puzzles (if LLM attempted something else)
    end_id = room.get("graph", {}).get("end")
    node_ids = {n.get("id") for n in room.get("graph", {}).get("nodes", [])}
    puzzle_ids = [p["id"] for p in room.get("puzzles", [])]
    if end_id not in node_ids:
        room.setdefault("graph", {}).setdefault("nodes", []).append(
            {"id": "final_door", "type": "gate", "requires": puzzle_ids}
        )
        room["graph"]["end"] = "final_door"
    else:
        # patch requires to include all puzzle ids
        for n in room["graph"]["nodes"]:
            if n.get("id") == end_id:
                n["requires"] = puzzle_ids

    # Validate; if too easy, harden; revalidate
    room = validate_room(room)
    if too_easy(room):
        room = harden(room, rng)
        room = validate_room(room)

    # Similarity guard; if too similar, perturb by regenerating offline wrapper title/intro
    if is_too_similar_to_recent(room, RECENT_WINDOW_DAYS):
        # tweak title/intro deterministically
        t2, i2 = rng.choice(THEMES)
        room["title"] = t2
        room["intro"] = i2

    # Optional critic refinement
    room = llm_critic_patch(room)
    room = validate_room(room)
    return room


# ---------- Public API: ensure/caching ----------

def ensure_daily_room(date_key: Optional[str] = None, force_regen: bool = False) -> EscapeRoom:
    """
    Get today’s room or generate and cache it.
    """
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
            current_app.logger.warning(f"[escape] generation attempt {attempt} failed: {e}")

    # If repeated failures, raise the last error (so you notice in logs)
    raise RuntimeError(f"Failed to generate room after {MAX_GEN_ATTEMPTS} attempts: {last_error}")


# ---------- Verify answers (server-side truth) ----------

def _match_answer(expected: str, submitted: str, pattern: Optional[str]) -> bool:
    """
    Compare with normalization; pattern-limited if specified.
    """
    if submitted is None:
        return False
    s = submitted.strip()
    e = str(expected)
    # If numeric pattern, require exact format
    if pattern and re.fullmatch(r"^\^\d+\$$", pattern):
        return s == e
    # Alphanumeric: compare normalized uppercase without spaces/punct
    return normalize_answer(s) == normalize_answer(e)


def verify_puzzle(room_json: Dict[str, Any], puzzle_id: str, answer: str) -> bool:
    """
    Routes will use this to check a submitted answer.
    """
    puzzles = room_json.get("puzzles", [])
    for p in puzzles:
        if p.get("id") == puzzle_id:
            sol = (p.get("solution") or {}).get("answer")
            pattern = (p.get("answer_format") or {}).get("pattern")
            if sol is None:
                return False
            return _match_answer(str(sol), str(answer or ""), pattern)
    return False


def verify_meta_final(room_json: Dict[str, Any], submitted: str) -> bool:
    """
    If you enable a meta final code (created during harden()), verify here.
    """
    expect = room_json.get("final_code")
    if not expect:
        return False
    return normalize_answer(submitted) == normalize_answer(expect)


# ---------- Scheduler hook ----------

_scheduler_started = False

def schedule_daily_generation(app) -> None:
    """
    Start APScheduler to generate the daily room just after local midnight.
    Safe to call from your app factory; idempotent within a process.
    """
    global _scheduler_started
    if _scheduler_started:
        current_app.logger.info("[escape] scheduler already started; skipping.")
        return
    _scheduler_started = True

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as e:
        current_app.logger.warning(f"[escape] APScheduler not available: {e}")
        return

    tz = pytz.timezone(TIMEZONE)
    scheduler = BackgroundScheduler(timezone=tz)

    def job():
        with app.app_context():
            date_key = get_today_key()
            try:
                ensure_daily_room(date_key)
                current_app.logger.info(f"[escape] Daily room generated for {date_key}")
            except Exception as e:
                current_app.logger.error(f"[escape] Daily generation failed for {date_key}: {e}")

    # Generate at 00:05 local time
    scheduler.add_job(job, "cron", hour=0, minute=5, id="escape_daily_gen", replace_existing=True)
    scheduler.start()
    current_app.logger.info("[escape] scheduler started (daily 00:05 local).")
