# -*- coding: utf-8 -*-
"""
apply_trailroom_patches.py  — robust patcher for your original core.py

What it does:
1) Adds three original mini-generators:
   - gen_vault_frenzy
   - gen_light_reactor
   - gen_pressure_chamber
2) Replaces gen_scene_mini() body to prefer those minis (legacy fallbacks kept)
3) Updates ensure_daily_room() to mix a per-request salt from Flask (if present):
   request.args['salt'] or request.args['rotate'].
   → This makes /escape/api/admin/regen?date=...&salt=v2 produce a new variant
     for the SAME date (fixes "Answer recently used").

Safe to re-run; creates a timestamped backup: core.py.bak_YYYYmmdd_HHMMSS

Usage:
  python -m app.escape.apply_trailroom_patches --dry-run
  python -m app.escape.apply_trailroom_patches
"""

from __future__ import annotations
import argparse
import datetime as dt
import re
from pathlib import Path

# ───────────────────────── Patch payloads ─────────────────────────

MINI_FUNCS_BLOCK = r'''
# ===== BEGIN CHATGPT MINI-GAMES (do not edit) =====
def gen_vault_frenzy(rng, pid, blacklist, theme=""):
    """
    Vault Frenzy — “pop-lock windows”
    Deterministic sequence mini: players reproduce the action series using control chips.
    Server verifies exact token sequence (comma-separated).
    """
    actions = ["tap", "tap", "hold", "left", "right", "tap", "up", "down"]
    L = rng.randint(8, 12)
    seq = [rng.choice(actions) for _ in range(L)]
    prompt = (
        f"{theme or 'The Salt Vault'}: Lock pins blink in waves. "
        "Hit the locks as they pop. Use the controls below; reproduce the full action series."
    )
    return Puzzle(
        id=pid,
        archetype="mini",
        mechanic="sequence_input",
        prompt=prompt,
        answer_format={"pattern": r"^(?:[A-Za-z_]+(?:,\\s*)?){5,}$"},
        solution={"answer": ",".join(seq)},
        hints=[
            "Watch the glow rhythm; it loops.",
            f"The series is {L} steps long.",
        ],
        decoys=["tap,hold,tap,left,right", "tap,tap,hold,right,left,up,down"],
        paraphrases=[
            "The lock pops in a fixed pattern—repeat it with the chips.",
            "Recreate the popping sequence using the control tokens."
        ],
        ui_spec={
            "sequence": ["tap", "hold", "left", "right", "up", "down"],
            "cue_set": {"style": "glow_windows", "tempo_ms": 550}
        }
    )

def _letter_filler(rng, avoid=""):
    A = [c for c in string.ascii_uppercase if c not in set((avoid or "").upper())]
    return rng.choice(A) if A else rng.choice(string.ascii_uppercase)

def _grid_with_path_for_word(rng, word, side=4):
    """
    Place `word` as an adjacent path on a side×side grid; fill the rest with random letters.
    Returns (grid, path_coords).
    """
    w = re.sub(r"[^A-Za-z]", "", word or "").upper()
    side = max(3, min(6, int(side or 4)))
    r, c = rng.randrange(side), rng.randrange(side)
    grid = [[None for _ in range(side)] for _ in range(side)]
    grid[r][c] = w[0]
    path = [(r, c)]
    for ch in w[1:]:
        dirs = rng.sample([(1,0),(-1,0),(0,1),(0,-1)], k=4)
        placed = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < side and 0 <= nc < side and grid[nr][nc] is None:
                grid[nr][nc] = ch
                r, c = nr, nc
                path.append((r, c))
                placed = True
                break
        if not placed:
            empties = [(i,j) for i in range(side) for j in range(side) if grid[i][j] is None]
            if not empties:
                break
            r, c = rng.choice(empties)
            grid[r][c] = ch
            path.append((r, c))
    for i in range(side):
        for j in range(side):
            if grid[i][j] is None:
                grid[i][j] = _letter_filler(rng, avoid=w)
    return grid, path

def gen_light_reactor(rng, pid, blacklist, theme=""):
    """
    Light Reactor — “chasing charge”
    Grid mini: collect letters by dragging a continuous path to spell a word.
    Server verifies the resulting string (letters joined without separators).
    """
    base_words = globals().get("ANAGRAM_WORDS", ["EMBER","CABLE","LUMEN","VAULT","PRESS","NOVA","CIRCUIT"])
    bl = set(x.lower() for x in (blacklist or set()))
    cands = [w for w in base_words if 4 <= len(w) <= 6 and w.lower() not in bl]
    ans = (rng.choice(cands) if cands else "EMBER")
    ans = re.sub(r"[^A-Za-z]", "", ans).upper()
    side = 4 if len(ans) <= 5 else 5
    grid, path = _grid_with_path_for_word(rng, ans, side=side)
    prompt = (
        f"{theme or 'Signal in the Sublevel'}: Capacitors pulse on a board. "
        "Drag through adjacent tiles to siphon charge—collect letters that spell the word."
    )
    return Puzzle(
        id=pid,
        archetype="mini",
        mechanic="grid_input",
        prompt=prompt,
        answer_format={"pattern": r"^[A-Za-z]{4,12}$"},
        solution={"answer": ans},
        hints=[
            "Start at the brightest cell.",
            "Adjacent moves only (no diagonals)."
        ],
        decoys=[ans[::-1], re.sub(r".$", "", ans), ans[1:] + ans[:1]],
        paraphrases=[
            "Trace a continuous path to pick up letters in order.",
            "Follow the energy trail; it spells a word."
        ],
        ui_spec={
            "grid": grid,
            "start": {"row": int(path[0][0]), "col": int(path[0][1])},
            "collect": "letters"
        }
    )

def gen_pressure_chamber(rng, pid, blacklist, theme=""):
    """
    Pressure Chamber — “dial & relief valves”
    Sequence mini: adjust pressure using control tokens; exact sequence is checked server-side.
    """
    tokens = ["tap", "hold", "up", "down", "left", "right"]
    L = rng.randint(7, 10)
    seq = []
    pressure = 0
    for _ in range(L):
        t = rng.choice(tokens)
        seq.append(t)
        if t in ("up", "right"): pressure += 1
        elif t in ("down", "left"): pressure -= 1
        else: pressure += 0
    prompt = (
        f"{theme or 'Archive of Ash'}: Gauges rattle; a relief valve hisses. "
        "Stabilize the chamber by repeating the control sequence."
    )
    return Puzzle(
        id=pid,
        archetype="mini",
        mechanic="sequence_input",
        prompt=prompt,
        answer_format={"pattern": r"^(?:[A-Za-z_]+(?:,\\s*)?){5,}$"},
        solution={"answer": ",".join(seq), "target_pressure": pressure},
        hints=[
            "The controls repeat in a loop.",
            "Think short bursts—then holds."
        ],
        decoys=["tap,up,tap,down,hold,right,left", "hold,hold,up,down,left,right,tap"],
        paraphrases=[
            "Repeat the valve/dial moves exactly.",
            "Match the stabilization inputs in order."
        ],
        ui_spec={"sequence": ["tap","hold","up","down","left","right"]}
    )
# ===== END CHATGPT MINI-GAMES =====
'''.lstrip("\n")

NEW_GEN_SCENE_MINI = r'''
def gen_scene_mini(rng, pid, blacklist, theme=""):
    """
    Produce a *mini-game* puzzle spec.
    Bias heavily toward the three original minis, keep a couple of legacy generators as low-weight fallbacks.
    """
    choices = [
        (lambda: gen_vault_frenzy(rng, pid, blacklist, theme),        6),
        (lambda: gen_light_reactor(rng, pid, blacklist, theme),       6),
        (lambda: gen_pressure_chamber(rng, pid, blacklist, theme),    6),
        # fallbacks (low weight) — these exist in your original core.py:
        (lambda: gen_knightword(rng, pid, blacklist, theme),          1),
        (lambda: gen_pathcode(rng, pid, blacklist, theme),            1),
    ]
    funcs, weights = zip(*choices)
    f = rng.choices(funcs, weights=weights, k=1)[0]
    try:
        return f()
    except Exception:
        # Avoid hard failure; pick a safe fallback if something unexpected happens.
        try:
            return gen_pathcode(rng, pid, blacklist, theme)
        except Exception:
            return gen_knightword(rng, pid, blacklist, theme)
'''.lstrip("\n")

ENSURE_SALT_SNIPPET = r'''
    # Allow per-request variant via Flask query args (?salt=... or ?rotate=...)
    # so /escape/api/admin/regen?date=YYYY-MM-DD&salt=v2 rotates safely.
    try:
        from flask import has_request_context, request
        if has_request_context():
            _qs_salt = request.args.get("salt") or request.args.get("rotate")
            if _qs_salt:
                secret = f"{secret}::{_qs_salt}"
    except Exception:
        pass
'''

# ───────────────────────── Helpers ─────────────────────────

def _insert_before(src: str, marker_regex: str, payload: str) -> str:
    m = re.search(marker_regex, src, flags=re.M)
    if not m:
        raise RuntimeError(f"Marker not found for insertion: {marker_regex}")
    return src[:m.start()] + payload + "\n" + src[m.start():]

def _replace_function_body(src: str, func_name: str, new_func_block: str) -> str:
    """
    Replace the entire function block `def func_name(...): ...` up to the next top-level def/class or EOF.
    More robust than a single big regex; works with different signatures/whitespace.
    """
    # locate the start of the def line
    m = re.search(rf'(?m)^def\s+{re.escape(func_name)}\s*\(', src)
    if not m:
        raise RuntimeError(f"Could not find {func_name}()")
    start = m.start()

    # find the next top-level 'def ' or 'class ' AFTER start
    nxt = re.search(r'(?m)^(def|class)\s+\w', src[m.end():])
    end = (m.end() + nxt.start()) if nxt else len(src)

    # backtrack to the start of the line for 'def func'
    line_start = src.rfind("\n", 0, start) + 1
    # ensure we capture from line start to end
    old_block = src[line_start:end]

    # replace with new block; ensure trailing newline
    return src[:line_start] + new_func_block.rstrip() + "\n\n" + src[end:]

def _patch_ensure_daily_room_salt(src: str) -> str:
    """
    Insert ENSURE_SALT_SNIPPET right after the line where secret is read
    inside ensure_daily_room(...).
    """
    # locate ensure_daily_room block
    m = re.search(r'(?m)^def\s+ensure_daily_room\s*\(', src)
    if not m:
        raise RuntimeError("Could not find ensure_daily_room()")

    # From function start to the next top-level def/class or EOF
    nxt = re.search(r'(?m)^(def|class)\s+\w', src[m.end():])
    block_end = (m.end() + nxt.start()) if nxt else len(src)
    block = src[m.start():block_end]

    if "request.args.get(\"salt\")" in block or "ESCAPE_REGEN_SALT" in block:
        # already patched
        return src

    # Find the line where secret is loaded
    sec_line = re.search(r'(?m)^\s*secret\s*=\s*os\.getenv\(', block)
    if not sec_line:
        raise RuntimeError("Could not find 'secret = os.getenv(...)' in ensure_daily_room()")

    insert_at = m.start() + sec_line.end()
    # Insert the snippet AFTER that line
    # Find end-of-line after sec_line
    eol = src.find("\n", insert_at)
    if eol == -1:
        eol = insert_at
    return src[:eol+1] + ENSURE_SALT_SNIPPET + src[eol+1:]

# ───────────────────────── Patcher ─────────────────────────

def patch_core(core_path: Path, dry_run: bool = False) -> str:
    src = core_path.read_text(encoding="utf-8")
    actions = []

    # 1) Ensure we have random/string/re imports (your file already does, but keep it resilient)
    missing = []
    for mod in ("random", "string", "re"):
        if not re.search(rf'(?m)^\s*import\s+{mod}\b', src):
            missing.append(mod)
    if missing:
        # insert right after the first import line
        src = re.sub(r'(?m)^(import\s+[^\n]+)\n', lambda m: m.group(0) + f"import {', '.join(missing)}\n", src, count=1)
        actions.append(f"added imports: {', '.join(missing)}")

    # 2) Insert mini-game generators before gen_scene_mini if not already present
    if "BEGIN CHATGPT MINI-GAMES" not in src:
        src = _insert_before(src, r'(?m)^def\s+gen_scene_mini\s*\(', MINI_FUNCS_BLOCK)
        actions.append("inserted new mini-game generators")
    else:
        actions.append("mini-game generators already present (skipped)")

    # 3) Replace gen_scene_mini function body
    try:
        src = _replace_function_body(src, "gen_scene_mini", NEW_GEN_SCENE_MINI)
        actions.append("replaced gen_scene_mini()")
    except RuntimeError as e:
        raise RuntimeError(f"Could not replace gen_scene_mini(): {e}")

    # 4) Patch ensure_daily_room to read request salt (if not already patched)
    try:
        src = _patch_ensure_daily_room_salt(src)
        actions.append("patched ensure_daily_room() to accept ?salt=/&rotate=")
    except RuntimeError as e:
        actions.append(f"ensure_daily_room salt patch skipped: {e}")

    if dry_run:
        return "DRY-RUN (no write)\n" + "\n".join(f"- {a}" for a in actions)

    # 5) Backup + write
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = core_path.with_suffix(f".py.bak_{ts}")
    backup.write_text(core_path.read_text(encoding="utf-8"), encoding="utf-8")
    core_path.write_text(src, encoding="utf-8")
    return f"Patched {core_path}.\nBackup: {backup.name}\n" + "\n".join(f"- {a}" for a in actions)

# ───────────────────────── CLI ─────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch original core.py with new minis + per-request salt regen.")
    parser.add_argument("--core", type=str, default=None, help="Path to core.py (defaults to app/escape/core.py)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    core_path = Path(args.core) if args.core else (here / "core.py")
    if not core_path.exists():
        raise SystemExit(f"core.py not found at: {core_path}")

    print(patch_core(core_path, dry_run=args.dry_run))

if __name__ == "__main__":
    main()
