#!/usr/bin/env python3
import re, sys, os, shutil

PATCHES = []

def subn(label, s, pattern, repl_func, flags=0):
    rx = re.compile(pattern, flags)
    new_s, n = rx.subn(repl_func, s)
    PATCHES.append((label, n))
    return new_s

def patch_core(src):
    # 1) _make_sequence_mini pattern: allow underscores + longer length
    src = subn(
        "seq-mini pattern widen",
        src,
        r'("answer_format":\s*\{\s*"pattern":\s*r")\^\[A-Za-z0-9,\\-]{5,24}\$("}\s*,)',
        lambda m: f'{m.group(1)}^[A-Za-z0-9,_\\-]{{5,200}}${m.group(2)}',
        flags=re.DOTALL,
    )

    # 1b) gen_translate_with_legend: allow underscores
    src = subn(
        "legend pattern add underscore",
        src,
        r'("answer_format":\s*\{\s*"pattern":\s*r")\^\[A-Za-z0-9,\\-]{5,200}\$("}\s*,)',
        lambda m: f'{m.group(1)}^[A-Za-z0-9,_\\-]{{5,200}}${m.group(2)}',
        flags=re.DOTALL,
    )

    # 2) gen_numeric_lock wording: signed phrasing
    src = subn(
        "numeric_lock c2 wording",
        src,
        r'c2 = f"The third digit is the first digit plus \{d3 - d1\}\."',
        lambda m: '''delta2 = d3 - d1
    c2 = f"The third digit is the first digit {'plus ' + str(delta2) if delta2 >= 0 else 'minus ' + str(abs(delta2))}."''',
    )
    src = subn(
        "numeric_lock c3 wording",
        src,
        r'c3 = f"The last digit equals the second digit plus \{d4 - d2\}\."',
        lambda m: '''delta3 = d4 - d2
    c3 = f"The last digit equals the second digit {'plus ' + str(delta3) if delta3 >= 0 else 'minus ' + str(abs(delta3))}."''',
    )

    # 3) pathcode directions readability: “D, L, R, U”
    src = subn(
        "pathcode directions join with comma",
        src,
        r"Directions are \{\s*''\.join\(sorted\(set\(path\)\)\)\s*\}",
        lambda m: "Directions are {', '.join(sorted(set(path)))}",
    )

    # 4) Deduplicate legend/hint injection in _sanitize_trail_puzzles
    src = subn(
        "sanitize_trail_puzzles dedupe legend+upgrade",
        rsrc := src,
        r'((\n[ \t]*# Add scene-grounding legend[^\n]*\n[ \t]*if not .*?\n[ \t]*_inject_sequence_legend\(p, theme\)\n[ \t]*_upgrade_minigame_hints\(p\)\n){2})',
        lambda m: m.group(2),
        flags=re.DOTALL,
    )

    # 5a) Collapse double fix pair in _fixup_minigames
    src = subn(
        "fixup_minigames collapse double fix pair",
        src,
        r'(\n[ \t]*_force_pattern_match\(p\)\n[ \t]*_upgrade_minigame_hints\(p\))\s*\1',
        lambda m: m.group(1),
        flags=re.DOTALL,
    )

    # 5b) Deduplicate duplicated comment block
    src = subn(
        "fixup_minigames dedupe duplicate comment block",
        src,
        r'(\n[ \t]*# Finally, force the pattern and hints\n[ \t]*_force_pattern_match\(p\)\n[ \t]*_upgrade_minigame_hints\(p\)\n[ \t]*rt\["puzzle"\] = p\n)\s*\1',
        lambda m: m.group(1),
        flags=re.DOTALL,
    )

    # 6) Comment fix: “2 routes” -> “3 routes” (simple string replace, no regex)
    before = "sanitize to 3 rooms / 2 routes per room"
    after  = "sanitize to 3 rooms / 3 routes per room"
    count = src.count(before)
    src = src.replace(before, after)
    PATCHES.append(("compose_trailroom comment fix", count))

    # 7) Dedupe duplicate “puzzles validation…” comment line
    src = subn(
        "dedupe puzzles validation comment",
        src,
        r'(\n[ \t]*# puzzles validation \+ consistent fragment per room[^\n]*\n)\s*\1',
        lambda m: m.group(1),
        flags=re.DOTALL,
    )

    return src

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 apply_trailroom_patches.py app/escape/core.py")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    bak = path + ".bak"
    shutil.copy2(path, bak)
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        patched = patch_core(original)
        if patched == original:
            print("No changes applied (file already up to date?).")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(patched)
            print(f"Patched successfully. Backup: {bak}")
        for label, n in PATCHES:
            print(f"{label}: {n} change(s)")
    except re.error as e:
        # restore on regex error
        shutil.copy2(bak, path)
        print(f"Patch failed, restored from backup. Reason: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
