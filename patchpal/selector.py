# patchpal/selector.py
from __future__ import annotations
import os, sys, re, json, time
from pathlib import Path
from typing import List, Dict, Any
import feedparser
import requests

# --- Ensure the repo root (which contains the 'app' package) is importable ---
ROOT = Path(__file__).resolve().parents[1]  # repo root (contains /app and /patchpal)
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
        print(f"[selector] Could not import main aggregator; will use fallback if needed: {e}")
        USE_MAIN = False

# ---------- Relevance helpers (Universal-first + optional Stack mode) ----------
UNIVERSAL_VENDORS = {
    "microsoft","windows","office","exchange","teams","edge",
    "apple","ios","macos","safari",
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

_WORDS = re.compile(r"[a-z0-9+.#/-]+")

def _text(item: Dict[str, Any]) -> str:
    return " ".join([
        str(item.get("title","")),
        str(item.get("summary","") or item.get("content","") or ""),
        str(item.get("vendor_guess","")),
    ]).lower()

def _tokens(s: str) -> set[str]:
    return set(_WORDS.findall(s))

def is_universal(item: Dict[str,Any]) -> bool:
    return bool(UNIVERSAL_VENDORS & _tokens(_text(item)))

def matches_stack(item: Dict[str,Any], tokens_csv: str | None) -> bool:
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

def is_exploited_or_high_epss(item: Dict[str,Any]) -> bool:
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = float(item.get("epss") or 0.0)
    return kev or epss >= 0.5  # conservative default

def pick_top_candidates(pool: List[Dict[str, Any]], n: int, ws) -> List[Dict[str, Any]]:
    """
    STRICT stack-first:
      - If mode=stack and stack_tokens set:
          1) all items matching the stack (in pool order)
          2) then high-signal (KEV or EPSS>=0.5)
          3) then universal vendors
          4) then anything left
      - Else (mode=universal): universal OR high-signal first, then the rest.
    """
    mode = (getattr(ws, "stack_mode", "universal") or "universal").lower()
    tokens = (getattr(ws, "stack_tokens", "") or "").strip()

    def uniq_add(dst: list, src: list):
        seen = {id(x) for x in dst}
        for it in src:
            # try a semantic key if present, fall back to object id
            key = it.get("id") or it.get("cve") or it.get("url") or it.get("title") or id(it)
            if key in seen:
                continue
            dst.append(it)
            seen.add(key)
            if len(dst) >= n:
                break
        return dst

    # --- universal mode unchanged ---
    if mode != "stack" or not tokens:
        primary = [it for it in pool if is_universal(it) or is_exploited_or_high_epss(it)]
        backup  = [it for it in pool if it not in primary]
        out: list[Dict[str, Any]] = []
        uniq_add(out, primary)
        if len(out) < n:
            uniq_add(out, backup)
        return out[:n]

    # --- strict stack-first ---
    stack_items   = [it for it in pool if matches_stack(it, tokens)]
    high_signal   = [it for it in pool if is_exploited_or_high_epss(it)]
    universal     = [it for it in pool if is_universal(it)]

    out: list[Dict[str, Any]] = []
    uniq_add(out, stack_items)
    if len(out) < n:
        uniq_add(out, [it for it in high_signal if it not in out])
    if len(out) < n:
        uniq_add(out, [it for it in universal if it not in out])
    if len(out) < n:
        uniq_add(out, pool)  # final fill, maintain pool ranking

    # helpful debug (shows in Render logs)
    try:
        print(f"[selector] mode=stack tokens='{tokens}' "
              f"stack={len(stack_items)} high={len(high_signal)} univ={len(universal)} chosen={len(out)}")
    except Exception:
        pass

    return out[:n]

# -------------------- Fallback pool builders ---------------------------------

FALLBACK_KEV_URL = os.getenv(
    "PATCHPAL_KEV_URL",
    # CISA KEV JSON (URL sometimes changes; we handle failures gracefully)
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
)

FALLBACK_FEEDS = [
    # Chrome releases (security updates are universal)
    "https://chromereleases.googleblog.com/atom.xml",
    # MSRC blog (general, sometimes security)
    "https://msrc.microsoft.com/blog/feed/",
    # CISA alerts
    "https://www.cisa.gov/uscert/ncas/alerts.xml",
]

def _safe_get_json(url: str, timeout: int = 10) -> dict | None:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.json()
    except Exception:
        return None
    return None

def _safe_parse_feed(url: str, timeout: int = 10):
    try:
        return feedparser.parse(url)
    except Exception:
        return {"entries": []}

def build_fallback_pool(max_items: int = 120, days: int = 14) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    now = time.time()
    cutoff = now - days * 86400

    # 1) CISA KEV
    kev = _safe_get_json(FALLBACK_KEV_URL)
    if kev and isinstance(kev, dict):
        for v in kev.get("vulnerabilities", []):
            try:
                # Try both keys, the schema has varied
                ts = v.get("dateAdded") or v.get("dateAddedToCatalog") or ""
                added = time.mktime(time.strptime(ts[:10], "%Y-%m-%d")) if ts else now
            except Exception:
                added = now
            if added < cutoff:
                continue

            cve = v.get("cveID") or v.get("cve") or ""
            vendor = (v.get("vendorProject") or "").strip()
            prod = (v.get("product") or "").strip()
            title = f"{cve} — {vendor} {prod}".strip(" —")
            out.append({
                "title": title,
                "summary": (v.get("shortDescription") or "").strip(),
                "kev": True,
                "vendor_guess": vendor.lower(),
                "link": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                "source": "CISA KEV",
            })

    # 2) A couple of vendor/news feeds (kept small on purpose)
    for url in FALLBACK_FEEDS:
        feed = _safe_parse_feed(url)
        for e in feed.get("entries", []):
            # accept recent-ish entries only
            try:
                updated = e.get("updated_parsed") or e.get("published_parsed")
                ts = time.mktime(updated) if updated else now
            except Exception:
                ts = now
            if ts < cutoff:
                continue
            title = (e.get("title") or "").strip()
            summary = (e.get("summary") or e.get("description") or "").strip()
            out.append({
                "title": title,
                "summary": summary,
                "kev": False,
                "vendor_guess": title.lower(),
                "link": e.get("link") or "",
                "source": url,
            })

    # Light dedupe by title
    seen = set()
    uniq: List[Dict[str,Any]] = []
    for it in out:
        t = it.get("title","").strip().lower()
        if t and t not in seen:
            seen.add(t)
            uniq.append(it)
        if len(uniq) >= max_items:
            break
    return uniq

# -------------------- Main function PatchPal calls ---------------------------

def topN_today(n: int = 5, ws=None) -> List[Dict[str,Any]]:
    """
    1) Try your main site's ranked pool (200).
    2) If empty, use fallback (CISA KEV + a couple feeds).
    3) Apply same-day uniqueness (if main selector exists).
    4) Relevance filter (universal or stack).
    """
    # 1) Ranked pool from main site
    pool: List[Dict[str,Any]] = []
    used_main = False
    if pick_top_cyber_items:
        try:
            pool = pick_top_cyber_items(n=200) or []
            used_main = bool(pool)
        except Exception as e:
            print(f"[selector] main pool error; will fallback: {e}")

    # 2) Fallback if needed
    if not pool:
        pool = build_fallback_pool(max_items=200, days=14)
        print(f"[selector] Using fallback pool: {len(pool)} items")

    if not pool:
        return []

    # 3) Same-day uniqueness (only if main function exists and we used main pool)
    if used_main and select_unique_for_today:
        try:
            pool = select_unique_for_today(pool, n=200)
        except Exception as e:
            print(f"[selector] uniqueness filter failed (continuing): {e}")

    # 4) Relevance filter
    ws = ws or type("W", (), {"stack_mode":"universal","stack_tokens":None})()
    return pick_top_candidates(pool, n, ws)

# Optional: tiny wrapper to annotate “FYI” when an item isn’t universal/high-signal
# patchpal/selector.py
def render_item_text(item: Dict[str, Any], idx: int, tone: str) -> str:
    """
    Produce Slack-safe mrkdwn. Guarantees a non-empty string and trims to Slack's limits.
    """
    try:
        from .utils import render_item_text_core  # your existing formatter (may be a no-op fallback)
        base = render_item_text_core(item, idx, tone)
    except Exception:
        base = None

    # Guarantee string
    if not isinstance(base, str):
        # If some formatter returned a dict or None, coerce to something readable
        title = str(item.get("title") or f"Item {idx}")
        summary = str(item.get("summary") or item.get("content") or "")
        base = f"*{idx}) {title}*\n{summary}".strip()

    base = base.strip()
    if not base:
        base = f"*{idx})* (no details)"

    # Slack section text hard limit ~3000 chars; stay safe
    if len(base) > 2900:
        base = base[:2900] + "…"

    # Add a subtle heads-up if it may be less broadly applicable
    if not (is_universal(item) or is_exploited_or_high_epss(item)):
        base += "\n_*FYI:* may not apply broadly; review relevance._"

    return base