# patchpal/selector.py
from __future__ import annotations
import os, sys, re, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests
import feedparser

# --- import path (repo root) -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Tunables ---------------------------------------------------------------
EPSS_THRESHOLD  = float(os.getenv("PATCHPAL_EPSS_THRESHOLD", "0.70"))
FALLBACK_DAYS   = int(os.getenv("PATCHPAL_FALLBACK_DAYS", "7"))
REQUEST_TIMEOUT = int(os.getenv("PATCHPAL_HTTP_TIMEOUT", "10"))
CISA_KEV_URL    = os.getenv(
    "PATCHPAL_KEV_URL",
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
)

BASE_FEEDS = [
    # Browsers & OS vendors
    "https://chromereleases.googleblog.com/atom.xml",
    "https://www.mozilla.org/en-US/security/advisories/feed/",
    "https://helpx.adobe.com/security/atom.xml",
    "https://www.openssl.org/news/news.rss",

    # Microsoft
    "https://msrc.microsoft.com/update-guide/rss",
    "https://msrc.microsoft.com/blog/feed/",

    # Infra / appliances
    "https://www.cisco.com/security/center/psirtrss20/CiscoSecurityAdvisory.xml",
    "https://www.vmware.com/security/advisories.xml",
    "https://advisories.fortinet.com/rss.xml",

    # Cloud
    "https://cloud.google.com/feeds/gcp-security-bulletins.xml",
    "https://aws.amazon.com/security/feed/",

    # Platforms / ecosystems
    "https://about.gitlab.com/security/advisories.xml",
    "https://groups.google.com/forum/feed/kubernetes-announce/msgs/rss_v2_0.xml",

    # Gov alerts
    "https://www.cisa.gov/uscert/ncas/alerts.xml",
]
EXTRA_FEEDS = [u.strip() for u in os.getenv("PATCHPAL_EXTRA_FEEDS", "").replace("\n"," ").split(" ") if u.strip()]
FALLBACK_FEEDS = BASE_FEEDS + EXTRA_FEEDS
# Drop pre-release noise unless explicitly allowed
SKIP_PRE_RELEASE = os.getenv("PATCHPAL_INCLUDE_PRE_RELEASE", "0").lower() not in ("1","true","yes")
NOISY_TITLES = re.compile(r"\b(dev|beta|canary|nightly|insider|preview)\b", re.I)


# --- Relevance --------------------------------------------------------------
UNIVERSAL_VENDORS = {
    "microsoft","windows","office","exchange","teams","edge",
    "apple","ios","macos","safari",
    "google","chrome","android",
    "adobe","acrobat","reader",
    "zoom","openssl",
}
STACK_MAP = {
    "windows":{"microsoft","windows"},
    "macos":{"apple","macos","mac os x"},
    "linux":{"linux","ubuntu","debian","rhel","centos","almalinux","suse"},
    "ios":{"apple","ios"},
    "android":{"android","google"},
    "chrome":{"chrome","google chrome"},
    "edge":{"edge","microsoft edge"},
    "firefox":{"firefox","mozilla"},
    "ms365":{"microsoft 365","office","exchange","sharepoint","teams","o365"},
    "adobe":{"adobe","acrobat","reader"},
    "zoom":{"zoom"},
    "openssl":{"openssl"},
    "nginx":{"nginx"},
    "apache":{"apache http server","apache httpd","httpd"},
    "postgres":{"postgres","postgresql"},
    "mysql":{"mysql","mariadb"},
    "sqlserver":{"sql server","mssql"},
    "aws":{"aws","amazon web services"},
    "azure":{"azure","microsoft azure"},
    "gcp":{"gcp","google cloud","google compute"},
    "cisco":{"cisco"},
    "fortinet":{"fortinet","fortigate"},
    "vmware":{"vmware"},
}

_WORDS   = re.compile(r"[a-z0-9+.#/-]+", re.I)
_CVE_RE  = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_STOP = {"cve", "update", "security", "advisory", "release", "dev", "desktop", "channel"}

def _normalize_title_for_key(title: str) -> str:
    """Strip CVE prefix, punctuation, collapse spaces, take first 5 significant tokens."""
    s = title or ""
    s = s.lower()
    s = re.sub(r"^cve-\d{4}-\d{4,7}\s*[—\-:]\s*", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks[:5]) or s.strip()

def _product_key(item: Dict[str, Any]) -> str:
    """Best-effort product grouping so we can collapse dup advisories for the same thing."""
    t = _text(item)
    title = (item.get("title") or "").lower()

    # Explicit products we see a lot
    if "citrix" in t and "session recording" in t:
        return "citrix-session-recording"
    if "google" in t and "chrome" in t:
        return "google-chrome"
    if "apple" in t and any(x in t for x in ("ios","ipados","macos","safari")):
        return "apple-ecosystem"
    if "git " in (" " + t) or title.startswith("git "):
        return "git-core"

    # Generic fallback from title
    return _normalize_title_for_key(item.get("title") or "")

def _strip_html(s: str | None) -> str:
    return "" if not s else _HTML_RE.sub("", s)

def _text(item: Dict[str, Any]) -> str:
    return " ".join([
        str(item.get("title","")),
        str(item.get("summary","") or item.get("content","") or ""),
        str(item.get("vendor_guess","")),
    ]).lower()

def _tokens(s: str) -> set[str]:
    return set(_WORDS.findall(s.lower()))

def _extract_cves(*chunks: str) -> list[str]:
    found = set()
    for ch in chunks or []:
        for c in _CVE_RE.findall(ch or ""):
            found.add(c.upper())
    return list(found)

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
    try:
        epss = float(item.get("epss") or 0.0)
    except Exception:
        epss = 0.0
    return kev or epss >= EPSS_THRESHOLD

def _key_for(it: Dict[str, Any]):
    return it.get("id") or it.get("cve") or it.get("url") or it.get("link") or it.get("title") or id(it)

def _uniq_add(dst: list, src: list, cap: int):
    seen = {_key_for(x) for x in dst}
    for it in src:
        k = _key_for(it)
        if k in seen:
            continue
        dst.append(it)
        seen.add(k)
        if len(dst) >= cap:
            break
    return dst

# --- Sources ----------------------------------------------------------------
def _get_json(url: str) -> dict | None:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

def _parse_feed(url: str):
    try:
        return feedparser.parse(url)
    except Exception:
        return {"entries": []}

def _load_kev_set() -> set[str]:
    kev_set: set[str] = set()
    kev = _get_json(CISA_KEV_URL)
    if isinstance(kev, dict):
        for v in kev.get("vulnerabilities", []):
            cve = (v.get("cveID") or v.get("cve") or "").upper()
            if cve:
                kev_set.add(cve)
    return kev_set

def _enrich_epss(items: List[Dict[str,Any]]) -> None:
    # Collect CVEs
    cves, seen = [], set()
    for it in items:
        for c in _extract_cves(it.get("title",""), it.get("summary","")):
            if c not in seen:
                seen.add(c); cves.append(c)
    if not cves:
        return

    # FIRST API ~200 CVEs per request
    for i in range(0, len(cves), 150):
        chunk = cves[i:i+150]
        try:
            url = "https://api.first.org/data/v1/epss?cve=" + ",".join(chunk)
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if not r.ok:
                continue
            data = r.json().get("data", [])
            score_map = {d["cve"].upper(): float(d.get("epss", 0.0)) for d in data}
            for it in items:
                for c in _extract_cves(it.get("title",""), it.get("summary","")):
                    if c in score_map:
                        it["epss"] = max(float(it.get("epss") or 0.0), score_map[c])
        except Exception:
            continue

def build_fallback_pool(max_items: int = 300, days: int = FALLBACK_DAYS) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    now = time.time()
    cutoff = now - days * 86400

    kev_set = _load_kev_set()

    # KEV as explicit items (recent only)
    kev_json = _get_json(CISA_KEV_URL)
    if isinstance(kev_json, dict):
        for v in kev_json.get("vulnerabilities", []):
            try:
                ts = (v.get("dateAdded") or v.get("dateAddedToCatalog") or "")[:10]
                added = time.mktime(time.strptime(ts, "%Y-%m-%d")) if ts else now
            except Exception:
                added = now
            if added < cutoff:
                continue
            cve = (v.get("cveID") or v.get("cve") or "").upper()
            vendor = (v.get("vendorProject") or "").strip()
            prod = (v.get("product") or "").strip()
            title = f"{cve} — {vendor} {prod}".strip(" —")
            out.append({
                "title": title,
                "summary": _strip_html(v.get("shortDescription") or ""),
                "kev": True,
                "vendor_guess": (vendor or prod or title).lower(),
                "link": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
                "source": "CISA KEV",
                "cve": cve,
            })

    # Vendor / ecosystem feeds
    for url in FALLBACK_FEEDS:
        feed = _parse_feed(url)
        for e in feed.get("entries", []):
            try:
                updated = e.get("updated_parsed") or e.get("published_parsed")
                ts = time.mktime(updated) if updated else now
            except Exception:
                ts = now
            if ts < cutoff:
                continue

            title = _strip_html((e.get("title") or "").strip())

            # ⬇️ skip pre-release chatter like Dev/Beta/Canary/Nightly/Insider/Preview
            if SKIP_PRE_RELEASE and NOISY_TITLES.search(title):
                continue

            summary = _strip_html((e.get("summary") or e.get("description") or "").strip())
            cves = _extract_cves(title, summary)
            kev_flag = any(c in kev_set for c in cves)

            out.append({
                "title": title,
                "summary": summary,
                "kev": kev_flag,
                "vendor_guess": title.lower(),
                "link": e.get("link") or "",
                "source": url,
                "cve": cves[0] if cves else None,
            })

    # De-dupe (prefer by CVE, then title)
    seen_cve, seen_title, uniq = set(), set(), []
    for it in out:
        cve = (it.get("cve") or "").upper()
        ttl = (it.get("title") or "").strip().lower()
        if cve:
            if cve in seen_cve:
                continue
            seen_cve.add(cve)
        else:
            if ttl in seen_title:
                continue
            seen_title.add(ttl)
        uniq.append(it)
        if len(uniq) >= max_items:
            break

    _enrich_epss(uniq)  # best-effort

    # --- Collapse by product (1 per product per day). Prefer KEV, then higher EPSS.
    by_product: Dict[str, Dict[str,Any]] = {}
    def _score(it: Dict[str,Any]) -> tuple:
        kev = 1 if it.get("kev") or it.get("known_exploited") else 0
        try:
            epss = float(it.get("epss") or 0.0)
        except Exception:
            epss = 0.0
        return (kev, epss)

    for it in uniq:
        key = _product_key(it)
        cur = by_product.get(key)
        if cur is None or _score(it) > _score(cur):
            by_product[key] = it

    collapsed = list(by_product.values())
    # keep original-ish ranking by sorting with score, then stable title
    collapsed.sort(key=lambda it: (_score(it), (it.get("title") or "")), reverse=True)

    return collapsed[:max_items]


# --- Rendering (OPS VOICE) ---------------------------------------------------
_BULLET = "•"

def _short(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return (s[: limit - 1] + "…") if len(s) > limit else s

def _badge_line(item: Dict[str, Any]) -> str:
    epss = 0.0
    try:
        epss = float(item.get("epss") or 0.0)
    except Exception:
        pass
    is_kev = bool(item.get("kev") or item.get("known_exploited"))
    sev = "HIGH" if is_kev or epss >= EPSS_THRESHOLD else "MEDIUM"
    emoji = ":rotating_light:" if sev == "HIGH" else ":warning:"
    bits = [f"{emoji} *{sev}*"]
    if is_kev:
        bits.append("KEV")
    if epss >= 0.01:  # hide meaningless zero
        bits.append(f"EPSS {epss:.2f}")
    return " • ".join(bits)

def _docs_links(item: Dict[str, Any]) -> str:
    links: list[tuple[str, str]] = []
    t = _text(item)
    cve = (item.get("cve") or "").upper()
    src = item.get("link")

    def add(url: str | None, label: str):
        if url:
            links.append((url, label))

    # Microsoft / Windows
    if any(k in t for k in ("microsoft","windows","edge","msrc","office","exchange","teams")):
        if cve:
            add(f"https://msrc.microsoft.com/update-guide/vulnerability/{cve}", "MSRC advisory")
        add("https://support.microsoft.com/help/4027667/windows-update", "Windows Update")
        add("https://learn.microsoft.com/windows-server/administration/windows-server-update-services/manage/approve-and-deploy-updates", "WSUS deploy")
        add("https://learn.microsoft.com/mem/intune/protect/windows-update-for-business-configure", "Intune deadlines")
    # Chrome
    if "chrome" in t:
        add("https://support.google.com/chrome/answer/95414", "Chrome: update")
        add("https://support.google.com/chrome/a/answer/9027636", "Admin: force update")
    # Apple
    if any(k in t for k in ("apple","ios","ipad","macos","safari")):
        add("https://support.apple.com/HT201222", "Update iPhone/iPad")
        add("https://support.apple.com/HT201541", "Update macOS")
    # Common vendors
    if "cisco" in t:
        add("https://www.cisco.com/c/en/us/support/docs/psirt.html", "Cisco PSIRT")
    if "vmware" in t:
        add("https://www.vmware.com/security/advisories.html", "VMware advisories")
    if "fortinet" in t or "fortigate" in t:
        add("https://www.fortiguard.com/psirt", "Fortinet PSIRT")
        # Git
    if "git " in (" " + t) or (item.get("title","").lower().startswith("git ")):
        add("https://git-scm.com/downloads", "Git downloads")
        add("https://github.com/git/git/tree/master/Documentation/RelNotes", "Git release notes")
    # Citrix
    if "citrix" in t:
        add("https://support.citrix.com/security-bulletins", "Citrix security bulletins")

    add(src, "Vendor notice")  # always include if present

    # Dedup + cap to 3
    out, seen = [], set()
    for u, label in links:
        if not u or u in seen:
            continue
        seen.add(u); out.append(f"<{u}|{label}>")
        if len(out) >= 3:
            break
    return " · ".join(out)

def _actions_and_verify(item: Dict[str, Any], tone: str) -> Tuple[list[str], list[str]]:
    t = _text(item)
    fix: list[str] = []
    verify: list[str] = []

    # CHROME FIRST (so the presence of the word 'Windows' in the blog text doesn't hijack it)
    if "chrome" in t:
        fix.append("Update Chrome to the latest stable.")
        if tone == "detailed":
            fix += ["Admin: force update via policy.", "Restart browser/devices if needed."]
            verify += ["chrome://version on sample endpoints shows latest build."]
        return fix, verify

    # APPLE next
    if any(k in t for k in ("apple","ios","ipad","macos","safari")):
        fix.append("Update iOS/iPadOS/macOS to the latest version.")
        if tone == "detailed":
            fix += ["MDM: push update and enforce restart."]
            verify += ["MDM inventory shows minimum OS version across devices."]
        return fix, verify

    # MICROSOFT after vendor-specific cases
    if any(k in t for k in ("microsoft","windows","edge","office","exchange","teams")):
        fix.append("Run Windows Update / deploy latest security updates.")
        if tone == "detailed":
            fix += ["WSUS/Intune: approve & force install; reboot if required."]
            verify += ["MDM/VA shows target KBs installed on all scoped devices."]
        return fix, verify

    # DEFAULT
    fix.append("Apply the vendor security update/hotfix.")
    if tone == "detailed":
        fix += ["Schedule maintenance; restart if needed."]
        verify += ["Service/app version matches vendor advisory; VA re-scan is clean."]
    return fix, verify

def render_item_text(item: Dict[str, Any], idx: int, tone: str) -> str:
    """
    Build multi-doc links here, then hand off to utils.render_item_text_core
    so all wording/voice lives in one place.
    """
    # make a shallow copy and inject the preformatted docs string
    docs_str = _docs_links(item)          # e.g. "<url|Label> · <url|Label>"
    it = dict(item)
    it["_docs"] = docs_str

    try:
        from .utils import render_item_text_core
        return render_item_text_core(it, idx, tone)
    except Exception:
        # ultra-simple fallback if utils import ever fails
        title   = str(it.get("title") or f"Item {idx}").strip()
        summary = _short(_strip_html(it.get("summary") or it.get("content") or ""), 220 if (tone or "simple")=="simple" else 420)
        badges  = _badge_line(it)
        lines = [f"*{idx}) {title}*", badges, f"*TL;DR:* {summary}"]
        if docs_str:
            lines.append(f"*Docs:* {docs_str}")
        txt = "\n".join([l for l in lines if l]).strip()
        return txt[:2900] + "…" if len(txt) > 2900 else txt

# --- Selection / Ranking -----------------------------------------------------
def pick_top_candidates(pool: List[Dict[str, Any]], n: int, ws) -> List[Dict[str, Any]]:
    mode   = (getattr(ws, "stack_mode", "universal") or "universal").lower()
    tokens = (getattr(ws, "stack_tokens", "") or "").strip()

    if mode != "stack" or not tokens:
        primary = [it for it in pool if is_universal(it) or is_exploited_or_high_epss(it)]
        out: list[Dict[str, Any]] = []
        _uniq_add(out, primary, n)
        if len(out) < n:
            _uniq_add(out, pool, n)
        return out[:n]

    stack_items = [it for it in pool if matches_stack(it, tokens)]
    high_signal = [it for it in pool if is_exploited_or_high_epss(it)]
    universal   = [it for it in pool if is_universal(it)]

    out: list[Dict[str, Any]] = []
    _uniq_add(out, stack_items, n)
    if len(out) < n: _uniq_add(out, high_signal, n)
    if len(out) < n: _uniq_add(out, universal, n)
    if len(out) < n: _uniq_add(out, pool, n)

    try:
        print(f"[selector] mode=stack tokens='{tokens}' stack={len(stack_items)} high={len(high_signal)} univ={len(universal)} chosen={len(out)}")
    except Exception:
        pass
    return out[:n]

# --- Public API --------------------------------------------------------------
def topN_today(n: int = 5, ws=None) -> List[Dict[str,Any]]:
    pool = build_fallback_pool(max_items=300, days=FALLBACK_DAYS)
    if not pool:
        return []
    ws = ws or type("W", (), {"stack_mode":"universal","stack_tokens":None})()
    return pick_top_candidates(pool, n, ws)
