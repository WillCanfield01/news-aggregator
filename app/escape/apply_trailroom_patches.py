# -*- coding: utf-8 -*-
"""
apply_trailroom_patches.py
Patches app/escape/core.py to add the three new original minigames and
bias gen_scene_mini() to select them, keeping legacy minis as fallbacks.

- Safe to re-run (idempotent).
- Makes a timestamped backup of core.py in the same directory.
- Use --dry-run to preview changes.

Usage (from repo root or from app/escape/):
    python -m app.escape.apply_trailroom_patches
    # or
    python app/escape/apply_trailroom_patches.py

Optional:
    python -m app.escape.apply_trailroom_patches --core path/to/core.py --dry-run
"""

from __future__ import annotations
import argparse
import datetime as dt
import re
from pathlib import Path

# ───────────────────────── Patch payloads ─────────────────────────

MINI_FUNCS_BLOCK = r'''
# ===== BEGIN CHATGPT MINI-GAMES (do not edit) =====
def gen_vault_frenzy(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
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

def _letter_filler(rng: random.Random, avoid: str = "") -> str:
    A = [c for c in string.ascii_uppercase if c not in set((avoid or "").upper())]
    return rng.choice(A) if A else rng.choice(string.ascii_uppercase)

def _grid_with_path_for_word(rng: random.Random, word: str, side: int = 4):
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

def gen_light_reactor(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    """
    Light Reactor — “chasing charge”
    Grid mini: collect letters by dragging a continuous path to spell a word.
    Server verifies the resulting string (letters joined without separators).
    """
    base_words = globals().get("ANAGRAM_WORDS", ["EMBER","CABLE","LUMEN","VAULT","PRESS","NOVA","CIRCUIT"])
    cands = [w for w in base_words if 4 <= len(w) <= 6 and w.lower() not in set(x.lower() for x in (blacklist or set()))]
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

def gen_pressure_chamber(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
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
def gen_scene_mini(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
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
    except Exception as e:
        # Avoid hard failure; pick a safe fallback if something unexpected happens.
        try:
            return gen_pathcode(rng, pid, blacklist, theme)
        except Exception:
            return gen_knightword(rng, pid, blacklist, theme)
'''.lstrip("\n")

# ───────────────────────── Patcher ─────────────────────────

def patch_core(core_path: Path, dry_run: bool = False) -> str:
    src = core_path.read_text(encoding="utf-8")

    actions = []

    # 0) Ensure imports that our functions rely on are present (string, re, random are already in your original file,
    # but this helps if they were moved).
    # Most originals had: `import os, hmac, hashlib, random, string, re, json`
    missing_imports = []
    for mod in ("random", "string", "re"):
        if not re.search(rf"\bimport\s+{mod}\b", src):
            missing_imports.append(mod)
    if missing_imports:
        # place after the main import bundle line
        src = re.sub(
            r"(?m)^(import\s+[^\n]+)\n",
            lambda m: m.group(0) + f"import {', '.join(missing_imports)}\n",
            src, count=1
        )
        actions.append(f"added imports: {', '.join(missing_imports)}")

    # 1) Insert the MINI_FUNCS_BLOCK if not already present
    if "BEGIN CHATGPT MINI-GAMES" not in src:
        # Insert just BEFORE the definition of gen_scene_mini (so helpers are in scope above it)
        m = re.search(r"(?m)^def\s+gen_scene_mini\s*\(", src)
        if not m:
            raise RuntimeError("Could not find gen_scene_mini() in core.py.")
        insert_at = m.start()
        src = src[:insert_at] + MINI_FUNCS_BLOCK + "\n" + src[insert_at:]
        actions.append("inserted new mini-game generators")
    else:
        actions.append("mini-game generators already present (skipped)")

    # 2) Replace gen_scene_mini body with the new weighted chooser
    #    We capture from the `def gen_scene_mini(...):` line up to next top-level `def` or EOF.
    pattern = re.compile(r"(?ms)^def\s+gen_scene_mini\s*\([^\)]*\):.*?(?=^\s*def\s+\w|\Z)")
    if "gen_vault_frenzy" not in pattern.search(src).group(0):
        src = pattern.sub(NEW_GEN_SCENE_MINI, src, count=1)
        actions.append("replaced gen_scene_mini()")
    else:
        actions.append("gen_scene_mini already prefers new minis (skipped)")

    if dry_run:
        return "DRY-RUN (no write)\n" + "\n".join(f"- {a}" for a in actions)

    # 3) Backup and write
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = core_path.with_suffix(f".py.bak_{ts}")
    backup.write_text(core_path.read_text(encoding="utf-8"), encoding="utf-8")
    core_path.write_text(src, encoding="utf-8")

    return f"Patched {core_path}.\nBackup: {backup.name}\n" + "\n".join(f"- {a}" for a in actions)

# ───────────────────────── CLI ─────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch Trailroom core.py with original minigames.")
    parser.add_argument("--core", type=str, default=None, help="Path to core.py (defaults to app/escape/core.py next to this file)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    core_path = Path(args.core) if args.core else (here / "core.py")

    if not core_path.exists():
        raise SystemExit(f"core.py not found at: {core_path}")

    result = patch_core(core_path, dry_run=args.dry_run)
    print(result)

if __name__ == "__main__":
    main()
