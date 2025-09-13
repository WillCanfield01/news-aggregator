#!/usr/bin/env python3
import re, sys, shutil

TARGET = r"_offline_trail"

INSERT_BLOCK = r'''
    # NEW: avoid cooldown collisions in offline flow
    try:
        room = _replace_recent_answers(room, rng)
    except Exception as e:
        try: current_app.logger.warning("[escape] offline replace_recent skipped: %s", e)
        except Exception: pass
'''

def apply(src: str) -> str:
    # First, try to insert right after the variety reshuffle
    pat1 = r'(room\s*=\s*_reshuffle_mechanics_for_variety\(room\)\s*\n)'
    if re.search(pat1, src):
        return re.sub(pat1, r'\1' + INSERT_BLOCK + '\n', src, count=1)

    # Fallback: insert right before validate_trailroom(room)
    pat2 = r'(\n[ \t]*room\s*=\s*validate_trailroom\(room\))'
    if re.search(pat2, src):
        return re.sub(pat2, INSERT_BLOCK + r'\1', src, count=1)

    raise RuntimeError("Could not find insertion point in _offline_trail.")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "app/escape/core.py"
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if TARGET not in src:
        raise SystemExit("Did not find _offline_trail in core.py")

    bak = path + ".bak_offline_recent"
    shutil.copy2(path, bak)
    try:
        out = apply(src)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Patched offline recent-answer fix. Backup: {bak}")
    except Exception as e:
        shutil.copy2(bak, path)
        print(f"Patch failed and file restored. Reason: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
