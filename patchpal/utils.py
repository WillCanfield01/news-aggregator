# patchpal/utils.py
from __future__ import annotations
import os, re, html
from typing import Dict, Any, List, Tuple

# ----------- hygiene / parsing ------------------------------------------------
TAG_RE          = re.compile(r"<[^>]+>")
BR_RE           = re.compile(r"(?i)<\s*br\s*/?>")
P_CLOSE_RE      = re.compile(r"(?i)</\s*p\s*>")
P_OPEN_RE       = re.compile(r"(?i)<\s*p(\s[^>]*)?>")
LI_OPEN_RE      = re.compile(r"(?i)<\s*li(\s[^>]*)?>")
LI_CLOSE_RE     = re.compile(r"(?i)</\s*li\s*>")
MULTISPACE_RE   = re.compile(r"[ \t]+")
NEWLINES_RE     = re.compile(r"\n{3,}")
URL_FINDER_RE   = re.compile(r"https?://[^\s>]+")
WORDS_RE        = re.compile(r"[A-Za-z0-9.+#/_-]+")
CVE_RE          = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)

EPSS_SIG_MIN = float(os.getenv("PATCHPAL_EPSS_SHOW_MIN", "0.05"))   # hide trivial EPSS
EPSS_HIGH    = float(os.getenv("PATCHPAL_EPSS_THRESHOLD", "0.70"))  # same as selector

def _strip_html(s: str | None) -> str:
    s = html.unescape((s or "").replace("\xa0", " "))
    s = BR_RE.sub("\n", s)
    s = P_CLOSE_RE.sub("\n", s)
    s = P_OPEN_RE.sub("", s)
    s = LI_CLOSE_RE.sub("", s)
    s = LI_OPEN_RE.sub("• ", s)
    s = TAG_RE.sub("", s)
    s = MULTISPACE_RE.sub(" ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = NEWLINES_RE.sub("\n\n", s)
    return s.strip()

def _short(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return (s[: limit - 1] + "…") if len(s) > limit else s

def _urls_from(item: Dict[str, Any]) -> List[str]:
    # Collect any reasonable doc / vendor URLs
    fields = [
        item.get("_docs"),                 # pre-computed list string (selector can pass)
        item.get("advisory_url"),
        item.get("vendor_url"),
        item.get("cisa_url"),
        item.get("kev_url"),
        item.get("url"),
        item.get("link"),
        item.get("source"),
        item.get("summary"),
    ]
    out, seen = [], set()
    for f in fields:
        if not f:
            continue
        for u in URL_FINDER_RE.findall(html.unescape(str(f))):
            if u not in seen:
                seen.add(u); out.append(u)
    return out

def _first_doc(item: Dict[str, Any]) -> str | None:
    urls = _urls_from(item)
    return urls[0] if urls else None

def _extract_cves(*chunks: str) -> List[str]:
    found = set()
    for ch in chunks:
        if not ch: continue
        for c in CVE_RE.findall(ch):
            found.add(c.upper())
    return list(found)

# ----------- severity / badges / SLA -----------------------------------------
def _epss(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("epss") or 0.0)
    except Exception:
        return 0.0

def _sev_tuple(item: Dict[str, Any]) -> Tuple[str, bool, float]:
    # returns (severity, is_kev, epss)
    kev  = bool(item.get("kev") or item.get("known_exploited"))
    epss = _epss(item)
    sev  = (item.get("severity") or "").upper()
    if not sev:
        sev = "HIGH" if kev or epss >= EPSS_HIGH else "MEDIUM"
    return sev, kev, epss

def _deadline(item: Dict[str, Any]) -> str:
    sev, kev, epss = _sev_tuple(item)
    if kev or epss >= EPSS_HIGH or sev == "CRITICAL":
        return "24–48h"
    if sev == "HIGH":
        return "3–5 days"
    if sev == "MEDIUM":
        return "next patch cycle"
    return "when practical"

def _badge_line(item: Dict[str, Any]) -> str:
    sev, kev, epss = _sev_tuple(item)
    emoji = ":rotating_light:" if sev in {"CRITICAL","HIGH"} else ":warning:"
    bits = [f"{emoji} *{sev}*"]
    if kev:
        bits.append("KEV")
    if epss >= EPSS_SIG_MIN:
        bits.append(f"EPSS {epss:.2f}")
    # tack on the SLA hint
    bits.append(f"• Patch by: { _deadline(item) }")
    # join with bullets but keep bold on sev only
    out = " • ".join(bits)
    return out

# ----------- impact / affected / actions / verify ----------------------------
KEYWORDS = [
    ("remote code execution", ["rce", "arbitrary code execution"]),
    ("privilege escalation", ["privilege escalation", "elevation of privilege"]),
    ("command injection", ["command injection"]),
    ("deserialization", ["deserialization"]),
    ("path traversal", ["path traversal", "directory traversal"]),
    ("auth bypass", ["authentication bypass", "bypass authentication"]),
    ("info leak", ["information disclosure"]),
]

def _impact_line(title: str, summary: str, kev: bool) -> str:
    t = f"{title}. {summary}".lower()
    hits = []
    for label, needles in KEYWORDS:
        if any(n in t for n in needles):
            hits.append(label)
    extras = []
    if "unauth" in t or "without authentication" in t:
        extras.append("unauthenticated")
    if "user interaction" in t or "social engineering" in t:
        extras.append("user interaction required")
    if kev:
        extras.append("exploited in the wild")
    if not hits:
        hits.append("security update available")
    return ", ".join(hits + extras)

def _affected_guess(title: str, summary: str) -> str:
    t = f"{title} {summary}".lower()
    nouns = []
    def add(s: str): 
        if s not in nouns: nouns.append(s)
    if any(k in t for k in ("windows", "server", "msrc", "exchange", "sharepoint")):
        add("Windows / Windows Server")
    if any(k in t for k in ("edge", "chrome", "firefox", "browser")):
        add("Browsers")
    if any(k in t for k in ("ios","ipados","macos","safari","apple")):
        add("Apple endpoints")
    if any(k in t for k in ("linux","ubuntu","debian","rhel","suse","centos","alma")):
        add("Linux servers")
    if any(k in t for k in ("vmware","esxi","vcenter")):
        add("VMware infra")
    if any(k in t for k in ("cisco","fortinet","fortigate","appliance")):
        add("Network appliances")
    if any(k in t for k in ("citrix")):
        add("Citrix servers")
    return ", ".join(nouns) or "Impacted products"

def _build_actions(title: str, summary: str, tone: str) -> List[str]:
    t = f"{title} {summary}".lower()
    if any(k in t for k in ("windows","microsoft","msrc","edge","exchange","sharepoint")):
        if tone == "detailed":
            return [
                "Run Windows Update on endpoints/servers; install all security updates.",
                "If managed: WSUS → approve & deploy, or Intune → set deadline and force restart.",
                "Reboot where required; re-scan and validate patch level.",
            ]
        return [
            "Deploy latest Windows security updates.",
            "WSUS/Intune: approve & push; reboot if needed.",
        ]
    if "chrome" in t:
        return ["Update Chrome to latest stable.", "Admins: enforce AutoUpdate; relaunch to complete."] if tone=="detailed" \
               else ["Update Chrome to latest stable."]
    if any(k in t for k in ("ios","ipados","macos","safari","apple")):
        return ["Update iOS/iPadOS/macOS to the latest version.",
                "MDM: push update and enforce restart."] if tone=="detailed" \
               else ["Update iOS/iPadOS/macOS to the latest version."]
    if "citrix" in t:
        return ["Apply the vendor hotfix to all Session Recording servers.",
                "Schedule maintenance: stop services, apply hotfix, restart, validate."] if tone=="detailed" \
               else ["Apply Citrix Session Recording hotfix across servers."]
    # Generic fallback
    return ["Apply the vendor security update/hotfix on affected systems."] + \
           (["Schedule maintenance; restart services; validate."] if tone=="detailed" else [])

def _build_verify(title: str, summary: str) -> List[str]:
    t = f"{title} {summary}".lower()
    if "chrome" in t:
        return ["chrome://version shows 'Up to date'.", "Managed: policy report shows AutoUpdate enabled."]
    if any(k in t for k in ("windows","microsoft","msrc","edge")):
        return ["Settings → Windows Update shows no pending updates.", "PowerShell: Get-HotFix or your VA tool reflects patched state."]
    if any(k in t for k in ("ios","ipados","macos","apple")):
        return ["Devices report the target OS version in MDM.", "MDM inventory shows no vulnerable builds."]
    if "citrix" in t:
        return ["Citrix build/patch level matches vendor advisory.", "Service health OK after restart."]
    return ["Validate version/build matches the vendor advisory.", "Re-scan with your VA tool."]

# ----------- main render -----------------------------------------------------
def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    """
    Build Slack mrkdwn for a single item in an 'ops voice'.
    Sections:
      - Title
      - Badges + SLA
      - Impact (why you should care)
      - Affected
      - Fix now (bullets)
      - Verify (only in detailed)
      - Docs (1–3 links if available)
    """
    title   = (item.get("title") or f"Item {idx}").strip()
    raw_sum = item.get("summary") or item.get("content") or ""
    summary = _strip_html(raw_sum)
    sev, kev, _ = _sev_tuple(item)

    # format + trim
    impact  = _short(_impact_line(title, summary, kev), 220 if tone=="simple" else 420)
    affected= _affected_guess(title, summary)
    badges  = _badge_line(item)

    # actions & verify
    actions = _build_actions(title, summary, tone)
    verify  = _build_verify(title, summary) if tone == "detailed" else []

    # docs: prefer selector-provided multi links via item["_docs"]; otherwise first URL
    docs_links = []
    if isinstance(item.get("_docs"), str) and item["_docs"].strip():
        docs_links = [item["_docs"].strip()]   # already formatted by selector
    else:
        doc = _first_doc(item)
        if doc:
            docs_links = [f"<{doc}|Vendor advisory>"]

    lines = [
        f"*{idx}) {title}*",
        badges,
        f"*Impact:* {impact}",
        f"*Affected:* {affected}",
        f"*Fix{' (step-by-step)' if tone=='detailed' else ''}:*",
        *[f"• {a}" for a in actions],
    ]
    if verify:
        lines += [f"*Verify:*", *[f"• {v}" for v in verify]]
    if docs_links:
        lines.append(f"*Docs:* " + " · ".join(docs_links))

    out = "\n".join(lines).strip()
    # Slack section text hard limit ~3000 chars; keep a margin
    if len(out) > 2900:
        out = out[:2900] + "…"
    return out
