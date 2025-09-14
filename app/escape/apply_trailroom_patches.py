#!/usr/bin/env python3
# app/escape/apply_valve_order_patch.py
import re, sys, shutil, pathlib

HERE = pathlib.Path(__file__).resolve().parent
CORE = HERE / "core.py"

def R(p): return pathlib.Path(p).read_text(encoding="utf-8")
def W(p,s): pathlib.Path(p).write_text(s, encoding="utf-8")

GEN_FN = r'''
def gen_valve_order(rng: random.Random, pid: str, blacklist: set, theme: str = "") -> Puzzle:
    """
    Valve Precedence: Deduce the unique order of valves A..F/G from "X before Y" clues.
    Submit as A,B,C,....
    """
    import itertools, string

    n = rng.randint(6, 7)
    labels = list(string.ascii_uppercase[:n])  # ["A"..]
    # Start from a hidden true order, then emit a subset of edges that still yields a unique topo order.
    true_order = labels[:]
    rng.shuffle(true_order)

    # Represent edges as pairs (u, v) meaning u before v.
    def is_valid_order(order, edges):
        pos = {ch: i for i, ch in enumerate(order)}
        return all(pos[u] < pos[v] for (u, v) in edges)

    # Count topological orders consistent with edges (brute force is fine for n<=7).
    def count_orders(edges):
        c = 0
        for perm in itertools.permutations(labels):
            if is_valid_order(perm, edges):
                c += 1
                if c > 1:  # early out
                    break
        return c

    # Start with a small edge set derived from the true order; add edges until the solution is unique.
    edges = []
    # begin with 2–3 spaced edges
    for i in range(n - 1):
        if rng.random() < 0.45:
            edges.append((true_order[i], true_order[i + 1]))
    if len(edges) < 2:
        edges = [(true_order[i], true_order[i + 1]) for i in range(0, n - 1, 2)][:2]

    # Add extra cross edges until uniqueness (exactly one consistent permutation)
    tries = 0
    while count_orders(edges) != 1 and tries < 200:
        tries += 1
        i, j = sorted(rng.sample(range(n), 2))
        if i == j: 
            continue
        u, v = true_order[i], true_order[j]
        if (u, v) not in edges and (v, u) not in edges:
            edges.append((u, v))

    # If still ambiguous (rare), fall back to emitting the full chain (guarantees uniqueness but still solvable).
    if count_orders(edges) != 1:
        edges = [(true_order[i], true_order[i + 1]) for i in range(n - 1)]

    # Build prompt text with readable bullets, and explicitly list the tokens so the sanitizer sees a “series”.
    bullets = "\n".join([f"- {u} before {v}" for (u, v) in edges])
    token_line = "Labels: " + ", ".join(labels)

    answer = ",".join(true_order)
    k = n

    # Decoys: reverse, rotation, and a swap of one adjacent pair
    rev  = list(reversed(true_order))
    rot  = true_order[1:] + true_order[:1]
    swap = true_order[:]
    if n >= 2:
        swap_idx = rng.randrange(0, n - 1)
        swap[swap_idx], swap[swap_idx + 1] = swap[swap_idx + 1], swap[swap_idx]

    return Puzzle(
        id=pid,
        archetype="mini",
        prompt=(f"A valve board in the {theme or 'room'} lists constraints. Determine the ONLY order that fits them all.\n"
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
'''

def inject_generator(src:str)->str:
    if re.search(r'\bdef\s+gen_valve_order\s*\(', src):
        return src
    # Insert before gen_translate_with_legend (stable anchor in your file)
    return re.sub(r'(?=\ndef\s+gen_translate_with_legend\s*\()',
                  "\n" + GEN_FN + "\n", src, count=1)

def guard_legend_inject(src:str)->str:
    # Make _inject_sequence_legend skip non-chip token sets (i.e., our A..F sequences).
    pattern = r'def _inject_sequence_legend\([^\)]*\):\n\s+if \(p\.get\("type"\) != "mini"\) or \(p\.get\("mechanic"\) != "sequence_input"\):\s*\n\s+    return'
    if not re.search(pattern, src):
        return src  # function form may differ; bail gracefully
    # Add a token guard right after the existing early return checks
    return re.sub(
        pattern,
        lambda m: m.group(0) + (
            '\n    tokens = (p.get("ui_spec") or {}).get("sequence") or []\n'
            '    # Only inject the chip legend for our standard chip set; skip custom token sets like ["A","B",...]\n'
            '    if tokens and not all(t in SEQ_TOKENS for t in tokens):\n'
            '        return'
        ),
        src, count=1
    )

def register_in_rotation(src:str)->str:
    # 1) Add it to gen_scene_mini choices (right after gen_knightword)
    src = re.sub(
        r'(choices\s*=\s*\[\s*\n\s*\(gen_knightword,\s*\d+\)\s*,\s*\n)',
        r'\1        (gen_valve_order,         3),\n',
        src, count=1
    )
    # 2) Downweight plain audio memory mini from 3 → 1 if present
    src = re.sub(r'\(gen_signal_translate,\s*3\)', r'(gen_signal_translate,     1)', src)
    return src

def main(path):
    path = pathlib.Path(path)
    core = R(path)
    bak = str(path) + ".bak_valveorder"
    shutil.copy2(path, bak)

    try:
        out = inject_generator(core)
        out = guard_legend_inject(out)
        out = register_in_rotation(out)
        W(path, out)
        print(f"Patched. Backup: {bak}")
    except Exception as e:
        shutil.copy2(bak, path)
        print(f"Patch failed, restored from backup. Reason: {e}")
        raise

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else str(CORE)
    main(p)
