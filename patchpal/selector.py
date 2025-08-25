# patchpal/selector.py
from __future__ import annotations
import os, sys, re
from pathlib import Path
from typing import List, Dict, Any

Item = Dict[str, Any]

# --- Ensure the repo root (which contains the 'app' package) is importable ---
ROOT = Path(__file__).resolve().parents[1]  # .../news-aggregator
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

USE_MAIN = os.getenv("PATCHPAL_USE_MAIN_POOL", "1").lower() not in ("0", "false", "no")

# Try to import your site's pool + same-day uniqueness
pick_top_cyber_items = None
select_unique_for_today = None
if USE_MAIN:
    try:
        from app.aggregator import pick_top_cyber_items, select_unique_for_today
        print("[selector] Using main site aggregator pool.")
    except Exception as e:
        print(f"[selector] Could not import main aggregator; falling back: {e}")
        USE_MAIN = False

# ---------- Relevance helpers (Universal-first + optional Stack mode) ----------
UNIVERSAL_VENDORS = {
    "microsoft","windows","office","exchange","teams","edge",
    "apple","ios","macos",
    "google","chrome","android",
    "adobe","acrobat","reader",
    "zoom","openssl",
}

STACK_MAP = {
    "windows":  {"microsoft","windows"},
    "macos":    {"apple","macos","mac os x"},
    "linux":    {"linux","ubuntu","debian","rhel","centos","almalinux","suse"},
    "ios":      {"apple","ios"},
    "android":  {"android","google"},
    "chrome":   {"chrome","google chrome"},
    "edge":     {"edge","microsoft edge"},
    "firefox":  {"firefox","mozilla"},
    "ms365":    {"microsoft 365","office","exchange","sharepoint","teams","o365"},
    "adobe":    {"adobe","acrobat","reader"},
    "zoom":     {"zoom"},
    "openssl":  {"openssl"},
    "nginx":    {"nginx"},
    "apache":   {"apache http server","apache httpd","httpd"},
    "postgres": {"postgres","postgresql"},
    "mysql":    {"mysql","mariadb"},
    "sqlserver":{"sql server","mssql"},
    "aws":      {"aws","amazon web services"},
    "azure":    {"azure","microsoft azure"},
    "gcp":      {"gcp","google cloud","google compute"},
    "cisco":    {"cisco"},
    "fortinet": {"fortinet","fortigate"},
    "vmware":   {"vmware"},
}

# include underscore so tokens like "ms365" or "mac_os" don't split oddly
_WORDS = re.compile(r"[a-z0-9+._#/-]+")

def _text(item: Item) -> str:
    return " ".join([
        str(item.get("title","")),
        str(item.get("summary","") or item.get("content","") or ""),
        str(item.get("vendor_guess","")),
    ]).lower()

def _tokens(s: str) -> set[str]:
    return set(_WORDS.findall(s))

def is_universal(item: Item) -> bool:
    return bool(UNIVERSAL_VENDORS & _tokens(_text(item)))

def matches_stack(item: Item, tokens_csv: str | None) -> bool:
    if not tokens_csv:
        return False
    chosen = [t.strip().lower() for t in tokens_csv.split(",") if t.strip()]
    if not chosen:
        return False
    t = _tokens(_text(item))
    for tok in chosen:
        for key in STACK_MAP.get(tok, set()):
            if key in t:
                return True
    return False

def _as_float(x) -> float:
    """Robust float parse for EPSS-like values; accepts '0.73', 0.73, or '73%'."""
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x[:-1]) / 100.0
        return float(x)
    except Exception:
        return 0.0

def is_exploited_or_high_epss(item: Item) -> bool:
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = _as_float(item.get("epss"))
    return kev or epss >= 0.5  # conservative default

def pick_top_candidates(pool: List[Item], n: int, ws) -> List[Item]:
    """Universal-first; if mode=='stack', require stack OR KEV/high-EPSS; fill from backup."""
    primary: List[Item] = []
    backup: List[Item] = []

    for it in pool:
        ok_universal = is_universal(it)
        ok_exploit   = is_exploited_or_high_epss(it)
        ok_stack     = matches_stack(it, getattr(ws, "stack_tokens", None))

        if (getattr(ws, "stack_mode", None) or "universal") == "stack":
            if ok_stack or ok_exploit or ok_universal:
                primary.append(it)
            else:
                backup.append(it)
        else:
            if ok_universal or ok_exploit:
                primary.append(it)
            else:
                backup.append(it)

    out = primary[:n]
    if len(out) < n:
        out += backup[: (n - len(out))]
    return out[:n]

# -------------------- Main function PatchPal calls ---------------------------

def topN_today(n: int = 5, ws=None) -> List[Item]:
    """
    Build large ranked pool from therealroundup (if available),
    enforce same-day uniqueness, then apply relevance filter for this workspace.
    """
    # 1) Ranked pool
    pool: List[Item] = []
    if USE_MAIN and callable(pick_top_cyber_items):
        pool = pick_top_cyber_items(n=200)  # your main pipeline
    else:
        # Fallback: empty pool (so we don't post junk). Plug a local pool here if desired.
        pool = []

    if not pool:
        return []

    # 2) Same-day uniqueness (your main rule set)
    if USE_MAIN and callable(select_unique_for_today):
        pool = select_unique_for_today(pool, n=200)

    # 3) Relevance filter
    ws = ws or type("W", (), {"stack_mode":"universal","stack_tokens":None})()
    return pick_top_candidates(pool, n, ws)

# Optional: tiny wrapper to annotate “FYI” when an item isn’t universal/high-signal
def render_item_text(item: Item, idx: int, tone: str) -> str:
    from .utils import render_item_text_core  # reuse your existing text logic
    base = render_item_text_core(item, idx, tone)
    if not (is_universal(item) or is_exploited_or_high_epss(item)):
        base += "\n_*FYI:* may not apply broadly; review relevance._"
    return base
