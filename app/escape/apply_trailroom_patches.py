#!/usr/bin/env python3
import re, sys, shutil, io

def patch_core(src: str) -> str:
    # Insert a pre-validation hardening block right before 'room = validate_trailroom(blob)'
    pattern = r'(\n[ \t]*room\s*=\s*validate_trailroom\(blob\))'
    inject = r'''
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
    '''
    return re.sub(pattern, inject, src, count=1)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "app/escape/core.py"
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    bak = path + ".bak2"
    shutil.copy2(path, bak)
    try:
        out = patch_core(src)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Applied generation hardening. Backup: {bak}")
    except Exception as e:
        shutil.copy2(bak, path)
        print(f"Patch failed; restored backup. Reason: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
