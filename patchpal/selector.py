# patchpal/selector.py
from __future__ import annotations
import os, sys, re, time, html, json, pathlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from slack_sdk.errors import SlackApiError  # add near the top with other imports
import requests
import feedparser
from .storage import recent_keys as _db_recent_keys, remember_posts as _db_remember_posts
# use DB-backed token store
from .install_store import get_bot_token
import random
# --- import path (repo root) -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Tunables ---------------------------------------------------------------
EPSS_THRESHOLD  = float(os.getenv("PATCHPAL_EPSS_THRESHOLD", "0.70"))
FALLBACK_DAYS   = int(os.getenv("PATCHPAL_FALLBACK_DAYS", "3"))   # fresher pool by default
REQUEST_TIMEOUT = int(os.getenv("PATCHPAL_HTTP_TIMEOUT", "10"))
CISA_KEV_URL    = os.getenv(
    "PATCHPAL_KEV_URL",
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
)
AI_SUMMARY_ON   = os.getenv("PATCHPAL_AI_REWRITE", "0").lower() in ("1","true","yes")
AI_MODEL        = os.getenv("PATCHPAL_AI_MODEL", "gpt-4o-mini")

# Variety controls
MAX_PER_VENDOR  = int(os.getenv("PATCHPAL_MAX_PER_VENDOR", "2"))

# Per-team dedupe memory
REMEMBER_DAYS   = int(os.getenv("PATCHPAL_REMEMBER_DAYS", "5"))

# Caching & speed controls
POOL_TTL_SEC    = int(os.getenv("PATCHPAL_POOL_TTL", "180"))  # reuse pool for ~3m
SKIP_EPSS       = os.getenv("PATCHPAL_SKIP_EPSS", "0").lower() in ("1", "true", "yes")

# Title style
SIMPLE_TITLES       = os.getenv("PATCHPAL_SIMPLE_TITLES", "1").lower() in ("1","true","yes")
SHOW_IDS_IN_TITLE   = os.getenv("PATCHPAL_SHOW_IDS", "0").lower() in ("1","true","yes")
TITLE_MAX_CHARS     = int(os.getenv("PATCHPAL_TITLE_MAX", "90"))
_POOL_CACHE: dict[tuple, tuple[float, List[Dict[str, Any]]]] = {}

# --- Feeds -------------------------------------------------------------------
BASE_FEEDS = [
    # Browsers & OS vendors
    "https://chromereleases.googleblog.com/atom.xml",
    "https://www.mozilla.org/en-US/security/advisories/feed/",
    "https://helpx.adobe.com/security/atom.xml",
    "https://www.openssl.org/news/news.rss",

    # Microsoft
    "https://msrc.microsoft.com/update-guide/rss",
    "https://msrc.microsoft.com/blog/feed/",

    # Network & appliances PSIRTs
    "https://www.cisco.com/security/center/psirtrss20/CiscoSecurityAdvisory.xml",
    "https://www.vmware.com/security/advisories.xml",
    "https://advisories.fortinet.com/rss.xml",
    "https://support.f5.com/csp/article/k9997/feed",                 # F5
    "https://support.citrix.com/security-bulletins/rss.xml",         # Citrix
    "https://kb.juniper.net/Feed?channel=SECURITY_ADVISORIES",       # Juniper
    "https://security.netapp.com/advisory/rss.xml",                  # NetApp
    "https://security.paloaltonetworks.com/rss.xml",                 # Palo Alto Networks

    # Cloud / platforms
    "https://cloud.google.com/feeds/gcp-security-bulletins.xml",
    "https://aws.amazon.com/security/feed/",
    "https://about.gitlab.com/security/advisories.xml",
    "https://groups.google.com/forum/feed/kubernetes-announce/msgs/rss_v2_0.xml",

    # Linux & distro security
    "https://usn.ubuntu.com/rss.xml",                                # Ubuntu USN
    "https://www.debian.org/security/dsa.rdf",                       # Debian DSA
    "https://www.suse.com/support/update/announcement/rss/",         # SUSE
    "https://lists.fedoraproject.org/archives/list/package-announce@lists.fedoraproject.org/latest.rss",
    "https://access.redhat.com/hydra/rest/securitydata/cvrf.xml",    # Red Hat (CVRF index)

    # Apps / ecosystems
    "https://www.jenkins.io/security/advisories/rss.xml",
    "https://www.drupal.org/security/rss.xml",
    "https://zerodayinitiative.com/rss/published/",

    # Mobile / Oracle CPU
    "https://source.android.com/security/bulletin/atom.xml",
    "https://www.oracle.com/security-alerts/rss-notifications.xml",

    # Gov alerts
    "https://www.cisa.gov/uscert/ncas/alerts.xml",
    "https://www.cisa.gov/uscert/ncas/current-activity.xml",
]
EXTRA_FEEDS = [u.strip() for u in os.getenv("PATCHPAL_EXTRA_FEEDS", "").replace("\n"," ").split(" ") if u.strip()]
FALLBACK_FEEDS = BASE_FEEDS + EXTRA_FEEDS

# Drop pre-release noise unless explicitly allowed
SKIP_PRE_RELEASE = os.getenv("PATCHPAL_INCLUDE_PRE_RELEASE", "0").lower() not in ("1","true","yes")
NOISY_TITLES = re.compile(r"\b(dev|beta|canary|nightly|insider|preview)\b", re.I)

# Drop think pieces (kept if clear CVE/patch signals)
SKIP_LOW_SIGNAL = os.getenv("PATCHPAL_SKIP_LOW_SIGNAL", "1").lower() not in ("0","false","no")
SIGNAL_RE = re.compile(
    r"(CVE-\d{4}-\d{4,7}|\badvisory\b|\bsecurity (update|fix|bulletin|patch)\b|\bKB\d{6,}\b|update guide|msrc)",
    re.I,
)
BLOGGY_RE = re.compile(
    r"(securing the ecosystem|best practices|what we learned|case study|our approach|defense in depth|bluehat)",
    re.I,
)

def _is_signal(title: str, summary: str, link: str, _source: str) -> bool:
    blob = " ".join([title or "", summary or "", link or ""])
    return bool(SIGNAL_RE.search(blob)) and not BLOGGY_RE.search(blob)

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

ALIASES = {
    "microsoft windows": "Windows",
    "microsoft edge": "Microsoft Edge",
    "citrix netscaler application delivery controller and gateway": "Citrix NetScaler",
    "citrix netscaler adc and gateway": "Citrix NetScaler",
    "citrix session recording": "Citrix Session Recording",
    "google chrome": "Chrome",
    "red hat enterprise linux": "Red Hat Enterprise Linux",
    "palo alto networks": "Palo Alto Networks",
    "qnap qhora-322": "QNAP QHora-322",
}

def _alias_product(name: str) -> str:
    s = name.lower()
    for k, v in ALIASES.items():
        if k in s:
            return v
    return name

# --- Shared regex / helpers --------------------------------------------------
_WORDS   = re.compile(r"[A-Za-z0-9.+#/_-]+")
_CVE_RE  = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_STOP    = {"cve", "update", "security", "advisory", "release", "dev", "desktop", "channel"}

BR_RE           = re.compile(r"(?i)<\s*br\s*/?>")
P_CLOSE_RE      = re.compile(r"(?i)</\s*p\s*>")
P_OPEN_RE       = re.compile(r"(?i)<\s*p(\s[^>]*)?>")
LI_OPEN_RE      = re.compile(r"(?i)<\s*li(\s[^>]*)?>")
LI_CLOSE_RE     = re.compile(r"(?i)</\s*li\s*>")
MULTISPACE_RE   = re.compile(r"[ \t]+")
NEWLINES_RE     = re.compile(r"\n{3,}")
VERSION_RE      = re.compile(r"(?<!CVE-)\b\d+(?:\.\d+){2,}\b")
LONG_DIGITS_RE  = re.compile(r"\b\d{6,}\b")
PAREN_NUM_RE    = re.compile(r"\((?:[^a-zA-Z]*\d[^)]*)\)")
SENT_SPLIT_RE   = re.compile(r"(?<=[\.\?!])\s+")
TITLE_DEDUP_RE  = re.compile(r"\b(\w+)\s+\1\b", re.I)
CVES_FIX_RE     = re.compile(r"\bCVEs-(\d{4}-\d{4,7})\b", re.I)
URL_FINDER_RE   = re.compile(r"https?://[^\s>]+'|https?://[^\s>]+")

BULLET = "•"

# ------------ html & text cleanup -------------
def _strip_html(s: str | None) -> str:
    s = html.unescape((s or "").replace("\xa0", " "))
    s = BR_RE.sub("\n", s)
    s = P_CLOSE_RE.sub("\n", s)
    s = P_OPEN_RE.sub("", s)
    s = LI_CLOSE_RE.sub("", s)
    s = LI_OPEN_RE.sub("• ", s)
    s = _HTML_RE.sub("", s)
    s = MULTISPACE_RE.sub(" ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = NEWLINES_RE.sub("\n\n", s)
    return s.strip()

def _remove_number_noise(s: str) -> str:
    s = VERSION_RE.sub("", s)
    s = LONG_DIGITS_RE.sub("", s)
    s = PAREN_NUM_RE.sub("", s)
    s = re.sub(r"\b(is|are)\s+being\s+rolled\s+out\b", "is rolling out", s, flags=re.I)
    s = re.sub(r"\bincluding:\s*", "Includes ", s, flags=re.I)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

PLAIN_MAP = [
    (re.compile(r"\bdeseriali[sz]ation of untrusted data\b", re.I), "unsafe handling of untrusted data"),
    (re.compile(r"\buse[- ]after[- ]free\b", re.I), "a memory bug attackers can exploit"),
    (re.compile(r"\binteger overflow\b", re.I), "a number-handling bug"),
    (re.compile(r"\bremote code execution\b|\bRCE\b", re.I), "letting attackers run code"),
    (re.compile(r"\bprivilege escalation\b|\belevation of privilege\b", re.I), "gaining more access than they should"),
    (re.compile(r"\bNetworkService\b", re.I), "a system service account"),
]
def _plain_english(s: str) -> str:
    t = s
    for rx, rep in PLAIN_MAP:
        t = rx.sub(rep, t)
    t = re.sub(r"\ba vulnerability\b", "a security bug", t, flags=re.I)
    t = re.sub(r"\bvulnerabilities\b", "security issues", t, flags=re.I)
    return t

def _tidy_title(s: str) -> str:
    s = CVES_FIX_RE.sub(r"CVE-\1", s)
    s = TITLE_DEDUP_RE.sub(r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip(" -–—")

def _normalize_title_for_key(title: str) -> str:
    s = title or ""
    s = s.lower()
    s = re.sub(r"^cve-\d{4}-\d{4,7}\s*[—\-:]\s*", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks[:5]) or s.strip()

def _text(item: Dict[str, Any]) -> str:
    return " ".join([
        str(item.get("title","")),
        str(item.get("summary","") or item.get("content","") or ""),
        str(item.get("vendor_guess","")),
    ]).lower()

def _tokens(s: str) -> set[str]:
    return set(_WORDS.findall(s.lower()))

def _product_key(item: Dict[str, Any]) -> str:
    t = _text(item)
    title = (item.get("title") or "").lower()
    if "citrix" in t and "session recording" in t:
        return "citrix-session-recording"
    if "google" in t and "chrome" in t:
        return "google-chrome"
    if "apple" in t and any(x in t for x in ("ios","ipados","macos","safari")):
        return "apple-ecosystem"
    if "git " in (" " + t) or title.startswith("git "):
        return "git-core"
    return _normalize_title_for_key(item.get("title") or "")

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
        r = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "PatchPal/1.0 (+https://therealroundup.com/patchpal)"},
        )
        if not r.ok:
            return {"entries": []}
        return feedparser.parse(r.content)
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
    if SKIP_EPSS:
        return
    cves, seen = [], set()
    for it in items:
        for c in _extract_cves(it.get("title",""), it.get("summary","")):
            if c not in seen:
                seen.add(c); cves.append(c)
    if not cves:
        return
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

# --- Extra-simple title humanizer -------------------------------------------
ZDI_ID_RE     = re.compile(r"\bZDI[-–—]?\s*\d{2,4}[-–—]?\d+\b", re.I)
CVE_IN_TITLE  = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)

def _clean_noise_segments(s: str) -> str:
    s = re.sub(r"[_/\\]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -–—,:;.")
    return s

def _guess_plain_label(text: str) -> str:
    t = text or ""
    if re.search(r"command injection", t, re.I):
        return "lets attackers run code"
    if re.search(r"\bremote code execution\b|\bRCE\b", t, re.I):
        return "lets attackers run code"
    if re.search(r"\b(privilege\s*(escalation|elevation))\b", t, re.I):
        return "lets attackers gain extra access"
    if re.search(r"\bauth(entication)?\s*bypass\b|login bypass", t, re.I):
        return "lets attackers bypass login"
    if re.search(r"\bdenial of service\b|\bDoS\b|\bcrash", t, re.I):
        return "can crash or stop the service"
    if re.search(r"\bsql injection\b", t, re.I):
        return "lets attackers inject database commands"
    if re.search(r"\bcross[- ]site scripting\b|\bXSS\b", t, re.I):
        return "injects scripts into web pages"
    if re.search(r"\b(info(?:rmation)? disclosure|data leak|exposure)\b", t, re.I):
        return "can leak data"
    return "security bug"

def _product_guess(raw_title: str) -> str:
    s = _clean_noise_segments(_strip_html(raw_title or ""))
    # Prefer the left side of a colon; otherwise the first clause before a dash
    if ":" in s:
        s = s.split(":", 1)[0]
    elif "—" in s or "-" in s:
        s = re.split(r"[—-]", s, 1)[0]
    # Trim obvious junk tokens
    s = re.sub(r"\b(RCE|XSS|DoS|CVE-\d{4}-\d{4,7}|ZDI[-–—]?\d{2,4}-\d+)\b.*", "", s, flags=re.I).strip()
    s = s or "This product"
    return _alias_product(s)

def _ids_from(raw_title: str) -> str:
    parts = []
    m1 = ZDI_ID_RE.search(raw_title or "")
    m2 = CVE_IN_TITLE.search(raw_title or "")
    if m1: parts.append(m1.group(0).upper().replace(" ", ""))
    if m2: parts.append(m2.group(0).upper())
    return " / ".join(dict.fromkeys(parts))  # de-dup while keeping order

def _human_title_very_simple(raw_title: str, context: str = "") -> str:
    """
    Build: '<Product>: <plain label>' and optionally add ' — <IDs>'.
    """
    product = _product_guess(raw_title)
    label   = _guess_plain_label((raw_title or "") + " " + (context or ""))
    headline = f"{product}: {label}"
    ids = _ids_from(raw_title)
    if SHOW_IDS_IN_TITLE and ids:
        headline = f"{headline} — {ids}"
    # hard cap
    if len(headline) > TITLE_MAX_CHARS:
        headline = headline[:TITLE_MAX_CHARS-1].rstrip() + "…"
    return headline

def build_fallback_pool(max_items: int = 300, days: int = FALLBACK_DAYS) -> List[Dict[str, Any]]:
    now = time.time()
    key = (days,)
    cached = _POOL_CACHE.get(key)
    if cached and (now - cached[0]) < POOL_TTL_SEC:
        # Serve from cache (copy then slice so callers can mutate safely)
        return cached[1][:max_items]

    out: List[Dict[str, Any]] = []
    cutoff = now - days * 86400
    kev_set = _load_kev_set()

    # ---- KEV (recent only) ----
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

    # ---- Vendor / ecosystem feeds ----
    for url in FALLBACK_FEEDS:
        feed = _parse_feed(url)
        for e in feed.get("entries", []):
            try:
                updated = e.get("updated_parsed") or e.get("published_parsed") or e.get("created_parsed")
                ts = time.mktime(updated) if updated else now
            except Exception:
                ts = now
            if ts < cutoff:
                continue

            title = _tidy_title(_strip_html(str(e.get("title") or "").strip())) or "(untitled)"

            entry_summary = ""
            try:
                content = e.get("content")
                if isinstance(content, list) and content:
                    entry_summary = content[0].get("value") or content[0].get("content") or ""
            except Exception:
                pass
            if not entry_summary:
                entry_summary = e.get("summary") or e.get("description") or ""
            entry_summary = _strip_html(str(entry_summary).strip())

            if SKIP_PRE_RELEASE and NOISY_TITLES.search(title):
                continue
            if SKIP_LOW_SIGNAL and not _is_signal(title, entry_summary, str(e.get("link") or ""), url):
                continue

            cves = _extract_cves(title, entry_summary)
            kev_flag = any(c in kev_set for c in cves)

            out.append({
                "title": title,
                "summary": entry_summary,
                "kev": kev_flag,
                "vendor_guess": (title + " " + entry_summary).lower(),
                "link": e.get("link") or "",
                "source": url,
                "cve": cves[0] if cves else None,
            })

    # ---- De-dupe by CVE/title ----
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

    _enrich_epss(uniq)

    # ---- Collapse by product, prefer KEV/EPSS ----
    def _score(it: Dict[str, Any]) -> Tuple[int, float]:
        kev = 1 if it.get("kev") or it.get("known_exploited") else 0
        try:
            epss = float(it.get("epss") or 0.0)
        except Exception:
            epss = 0.0
        return (kev, epss)

    by_product: Dict[str, Dict[str, Any]] = {}
    for it in uniq:
        key_prod = _product_key(it)
        cur = by_product.get(key_prod)
        if cur is None or _score(it) > _score(cur):
            by_product[key_prod] = it

    collapsed = list(by_product.values())
    collapsed.sort(key=lambda it: (_score(it), (it.get("title") or "")), reverse=True)

    # Back-fill to capacity
    chosen = {_key_for(it) for it in collapsed}
    for it in uniq:
        k = _key_for(it)
        if k in chosen:
            continue
        collapsed.append(it)
        chosen.add(k)
        if len(collapsed) >= max_items:
            break

    # Save full collapsed list to cache (not just the slice)
    _POOL_CACHE[key] = (time.time(), collapsed)
    return collapsed[:max_items]

# --- AI (summary-only) -------------------------------------------------------
def _ai_plain_summary(raw: str, max_sents: int) -> str | None:
    if not AI_SUMMARY_ON:
        return None
    try:
        from openai import OpenAI  # lazy import
    except Exception:
        return None
    try:
        client = OpenAI()
        sys_prompt = (
            f"Rewrite in plain English using up to {max_sents} short sentences. "
            "Avoid jargon and long numbers; keep it understandable to non-security readers."
        )
        resp = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0.2,
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":raw[:2000]},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

# --- Rendering ---------------------------------------------------------------
def _docs_links(item: Dict[str, Any]) -> str:
    links: list[tuple[str, str]] = []
    t = _text(item)
    cve = (item.get("cve") or "").upper()
    src = item.get("link")

    def add(url: str | None, label: str):
        if url:
            links.append((url, label))

    if any(k in t for k in ("microsoft","windows","edge","msrc","office","exchange","teams")):
        if cve:
            add(f"https://msrc.microsoft.com/update-guide/vulnerability/{cve}", "MSRC advisory")
        add("https://support.microsoft.com/help/4027667/windows-update", "Windows Update")
        add("https://learn.microsoft.com/windows-server/administration/windows-server-update-services/manage/approve-and-deploy-updates", "WSUS deploy")
        add("https://learn.microsoft.com/mem/intune/protect/windows-update-for-business-configure", "Intune deadlines")
    if "chrome" in t:
        add("https://support.google.com/chrome/answer/95414", "Chrome: update")
        add("https://support.google.com/chrome/a/answer/9027636", "Admin: force update")
    if any(k in t for k in ("apple","ios","ipad","macos","safari")):
        add("https://support.apple.com/HT201222", "Update iPhone/iPad")
        add("https://support.apple.com/HT201541", "Update macOS")
    if "cisco" in t:
        add("https://www.cisco.com/c/en/us/support/docs/psirt.html", "Cisco PSIRT")
    if "vmware" in t:
        add("https://www.vmware.com/security/advisories.html", "VMware advisories")
    if "fortinet" in t or "fortigate" in t:
        add("https://www.fortiguard.com/psirt", "Fortinet PSIRT")
    if "git " in (" " + t) or (item.get("title","").lower().startswith("git ")):
        add("https://git-scm.com/downloads", "Git downloads")
        add("https://github.com/git/git/tree/master/Documentation/RelNotes", "Git release notes")
    if "citrix" in t:
        add("https://support.citrix.com/security-bulletins", "Citrix security bulletins")

    add(src, "Vendor notice")

    out, seen = [], set()
    for u, label in links:
        if not u or u in seen:
            continue
        seen.add(u); out.append(f"<{u}|{label}>")
        if len(out) >= 3:
            break
    return " · ".join(out)

def _build_actions(title: str, summary: str, tone: str) -> Tuple[List[str], List[str]]:
    t = f"{title} {summary}".lower()
    fix: List[str] = []
    verify: List[str] = []

    if any(k in t for k in ("chrome","chromium","chromeos")):
        fix.append("Update Chrome/ChromeOS to the latest stable build.")
        if tone == "detailed":
            fix += ["Admins can force updates via policy.", "Restart browser/devices if needed."]
            verify += ["chrome://version shows the latest build on sample endpoints."]
        return fix, verify

    if any(k in t for k in ("apple","ios","ipados","macos","safari")):
        fix.append("Update iPhone/iPad/Mac to the latest version.")
        if tone == "detailed":
            fix += ["If managed, push the OS update via MDM and enforce a restart."]
            verify += ["MDM inventory shows the minimum OS version across devices."]
        return fix, verify

    if any(k in t for k in ("microsoft","windows","edge","office","exchange","teams")):
        fix.append("Run Windows Update or deploy the latest security updates.")
        if tone == "detailed":
            fix += ["WSUS/Intune: approve and force install; reboot if required."]
            verify += ["MDM/VA shows the target KBs installed across the fleet."]
        return fix, verify

    fix.append("Apply the vendor security update or hotfix.")
    if tone == "detailed":
        fix += ["Schedule maintenance; restart if needed."]
        verify += ["Version matches the advisory and a re-scan is clean."]
    return fix, verify

def _meta_lines(item: Dict[str, Any]) -> List[str]:
    kev = bool(item.get("kev") or item.get("known_exploited"))
    try:
        epss = float(item.get("epss") or 0.0)
    except Exception:
        epss = 0.0

    sev = (item.get("severity") or "").upper()
    if not sev:
        sev = "HIGH" if kev or epss >= EPSS_THRESHOLD else "MEDIUM"

    lines = [
        f"*Security alert:* {sev.title()}",
        f"*Known Exploited Vulnerability:* {'Yes' if kev else 'No'}",
    ]
    if epss >= 0.01:
        lines.append(f"*Exploit Prediction Scoring System:* {epss:.2f}")
    return lines

def _summary_text(title: str, summary: str, *, tone: str) -> str:
    raw = _strip_html(summary or title or "")
    raw = _plain_english(_remove_number_noise(raw))
    max_sents = 2 if (tone or "simple") == "simple" else 4
    if AI_SUMMARY_ON:
        ai = _ai_plain_summary(raw, max_sents)
        if ai:
            return ai if ai.endswith((".", "!", "?")) else (ai + ".")
    parts = [p.strip(" .;:-") for p in SENT_SPLIT_RE.split(raw) if p.strip()]
    if not parts:
        return title.strip() + "."
    text = ". ".join(parts[:max_sents]).strip(". ") + "."
    hard_cap = 420 if (tone or "simple") == "simple" else 800
    return (text[:hard_cap-1].rstrip()+"…") if len(text) > hard_cap else text

def _humanize_title(s: str) -> str:
    # safe fallback when SIMPLE_TITLES is disabled
    return _tidy_title(s or "")

def render_item_text(item: dict, idx: int, tone: str) -> str:
    raw_t = _tidy_title((item.get("title") or "").strip())
    summary_src = item.get("summary") or item.get("content") or ""
    title = _human_title_very_simple(raw_t, summary_src) if SIMPLE_TITLES else _humanize_title(raw_t)
    meta  = _meta_lines(item)
    summary = _summary_text(title, summary_src, tone=(tone or "simple"))
    fix, verify = _build_actions(title, summary, tone or "simple")
    actions_lines = [f"{BULLET} {a}" for a in fix]
    doc_str = _docs_links(item)

    blocks: List[str] = [
        f"*{idx}) {title}*",
        *meta,
        "",
        f"*Summary:* {summary}",
        "",
        "*Fix:*",
        *actions_lines,
    ]
    if (tone or "simple") == "detailed" and verify:
        blocks += ["", "*Verify:*", *[f"{BULLET} {v}" for v in verify]]

    if doc_str:
        blocks += ["", f"Docs: {doc_str}"]

    text = "\n".join(blocks).strip()
    return text[:2900] + "…" if len(text) > 2900 else text

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
    return out[:n]

# --- Recently posted memory (per team) ---------------------------------------

# --- Recently posted memory (per team) ---------------------------------------
def _recent_keys(team_id: str | None) -> set[str]:
    return _db_recent_keys(team_id or "", REMEMBER_DAYS) if team_id else set()

def mark_posted(team_id: str | None, items: List[Dict[str,Any]]):
    if team_id:
        _db_remember_posts(team_id, [str(_key_for(it)) for it in items])

# --- Public API --------------------------------------------------------------
def topN_today(n: int = 5, ws=None, team_id: str | None = None) -> List[Dict[str,Any]]:
    """Return up to N items for today, filtered by team’s recently-posted keys."""
    pool = build_fallback_pool(max_items=300, days=FALLBACK_DAYS)
    if len(pool) < n:
        pool = build_fallback_pool(max_items=300, days=FALLBACK_DAYS * 2)

    # dedupe by team memory
    tid = team_id or (getattr(ws, "team_id", None) if ws else None)
    seen = _recent_keys(tid) if tid else set()
    pool2 = [it for it in pool if str(_key_for(it)) not in seen] or pool

    ws = ws or type("W", (), {"stack_mode":"universal","stack_tokens":None})()
    picks = pick_top_candidates(pool2, n, ws)

    # vendor variety
    def _vendor_of(it: Dict[str,Any]) -> str:
        raw = (_text(it) or "").lower()
        if "red hat" in raw: return "redhat"
        if "palo alto" in raw: return "paloalto"
        if "google chrome" in raw or "chrome" in raw: return "chrome"
        for v in ["microsoft","apple","google","adobe","citrix","git","cisco","vmware",
                "fortinet","oracle","ubuntu","debian","suse","mozilla","f5","juniper",
                "netapp","jenkins","drupal","qnap","progress","atlassian"]:
            if v in raw:
                return v
        return "other"

    counts: Dict[str,int] = {}
    variety: List[Dict[str,Any]] = []
    extras: List[Dict[str,Any]] = []
    for it in picks:
        v = _vendor_of(it)
        if counts.get(v,0) < MAX_PER_VENDOR:
            variety.append(it); counts[v] = counts.get(v,0)+1
        else:
            extras.append(it)
    for it in extras:
        if len(variety) >= n:
            break
        variety.append(it)

    return variety[:n]

# --- Posting helper ----------------------------------------------------------
# --- Posting helper ----------------------------------------------------------
_LINK_RE = re.compile(r"<([^>|]+)\|([^>]+)>")
_TAG_RE  = re.compile(r"<[@#!][^>]+>")
_FMT_RE  = re.compile(r"[*_`~]")

def _fallback_text(md: str, limit: int = 300) -> str:
    if not isinstance(md, str):
        md = str(md or "")
    s = _LINK_RE.sub(r"\2", md)
    s = _TAG_RE.sub("", s)
    s = _FMT_RE.sub("", s)
    s = " ".join(s.split())
    return (s[:limit].rstrip() + "…") if len(s) > limit else s

def post_daily_digest(team_id: str, channel_id: str, tone: str = "simple"):
    from slack_sdk.web import WebClient  # lazy import
    token = get_bot_token(team_id)
    if not token:
        raise RuntimeError(f"No installation found for team {team_id}")
    client = WebClient(token=token)

    # 0) Sanity: token belongs to this team
    try:
        auth = client.auth_test()
        if auth.get("team_id") != team_id:
            print(f"[scheduler] auth mismatch for team {team_id}")
            return
    except SlackApiError as e:
        print(f"[scheduler] auth.test failed: {e.response.get('error')}")
        return

    # 1) Channel existence + membership
    try:
        info = client.conversations_info(channel=channel_id)
    except SlackApiError as e:
        print(f"[scheduler] conversations.info failed for {channel_id}: {e.response.get('error')}")
        return

    ch = info.get("channel") or {}
    if ch.get("is_archived"):
        print(f"[scheduler] channel archived: {channel_id}")
        return

    is_private = ch.get("is_private")
    is_member  = ch.get("is_member")

    if not is_member:
        if is_private:
            # cannot auto-join private channels
            print(f"[scheduler] not a member of private channel {channel_id} — invite the app")
            return
        # public: try to join
        try:
            client.conversations_join(channel=channel_id)
        except SlackApiError as e:
            print(f"[scheduler] join failed for {channel_id}: {e.response.get('error')}")
            return

    # 2) Build items (de-duped per team)
    ws_stub = type("W", (), {"team_id": team_id, "stack_mode": "universal", "stack_tokens": None})()
    items = topN_today(n=5, ws=ws_stub, team_id=team_id)
    if not items:
        print(f"[scheduler] no items to post for {team_id}")
        return

    # 3) Post header (rate-limit safe)
    try:
        hdr = _post_with_retry(
            client,
            channel=channel_id,
            text="Today’s Top 5 Patches / CVEs",
            blocks=[{
                "type": "header",
                "text": {"type": "plain_text", "text": "Today’s Top 5 Patches / CVEs", "emoji": True},
            }],
        )
    except SlackApiError as e:
        print(f"[scheduler] header post failed: {e.response.get('error')}")
        return

    parent_ts = hdr["ts"]

    # 4) Post each item as a thread reply (rate-limit safe)
    for idx, it in enumerate(items, 1):
        body = render_item_text(it, idx, tone) or f"{idx}) (no details)"
        try:
            _post_with_retry(
                client,
                channel=channel_id,
                thread_ts=parent_ts,
                text=_fallback_text(body, 300),
                blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": body}}],
            )
        except SlackApiError as e:
            print(f"[scheduler] item {idx} failed: {e.response.get('error')}")

    # mark items as posted only after we successfully post the header & loop
    mark_posted(team_id, items)

def _post_with_retry(client, **kwargs):
    """chat.postMessage with 429 handling + light jitter."""
    attempts = 5
    for i in range(attempts):
        try:
            return client.chat_postMessage(**kwargs)
        except SlackApiError as e:
            status = getattr(e.response, "status_code", None)
            err = (e.response.get("error") if hasattr(e, "response") else None) or ""
            if status == 429 or err in {"ratelimited", "rate_limited"}:
                retry = int(e.response.headers.get("Retry-After", "1")) if hasattr(e, "response") else 1
                sleep = retry + random.uniform(0, 0.5 * (i + 1))
                time.sleep(sleep)
                continue
            raise

           