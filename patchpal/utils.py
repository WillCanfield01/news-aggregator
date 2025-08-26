# patchpal/utils.py
from __future__ import annotations
import re, html, os
from typing import Dict, Any, List, Tuple

# ------------ regex helpers -------------
TAG_RE          = re.compile(r"<[^>]+>")
BR_RE           = re.compile(r"(?i)<\s*br\s*/?>")
P_CLOSE_RE      = re.compile(r"(?i)</\s*p\s*>")
P_OPEN_RE       = re.compile(r"(?i)<\s*p(\s[^>]*)?>")
LI_OPEN_RE      = re.compile(r"(?i)<\s*li(\s[^>]*)?>")
LI_CLOSE_RE     = re.compile(r"(?i)</\s*li\s*>")
MULTISPACE_RE   = re.compile(r"[ \t]+")
NEWLINES_RE     = re.compile(r"\n{3,}")

CVE_RE          = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
# Versions like 132.0.6834.241 or 10.15.7, but do not eat CVEs
VERSION_RE      = re.compile(r"(?<!CVE-)\b\d+(?:\.\d+){2,}\b")
LONG_DIGITS_RE  = re.compile(r"\b\d{6,}\b")                   # ids like 427681143
PAREN_NUM_RE    = re.compile(r"\((?:[^a-zA-Z]*\d[^)]*)\)")    # (Platform Version: 16093.115.0)
SENT_SPLIT_RE   = re.compile(r"(?<=[\.\?!])\s+")
WS_RE           = re.compile(r"\s+")
TITLE_DEDUP_RE  = re.compile(r"\b(\w+)\s+\1\b", re.I)          # "Git Git" -> "Git"
CVES_FIX_RE     = re.compile(r"\bCVEs-(\d{4}-\d{4,7})\b", re.I)# "CVEs-2024-1234" -> "CVE-2024-1234"

WORDS_RE        = re.compile(r"[A-Za-z0-9.+#/_-]+")
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
    s = TAG_RE.sub("", s)
    s = MULTISPACE_RE.sub(" ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = NEWLINES_RE.sub("\n\n", s)
    return s.strip()

def _pick_doc_url(item: Dict[str, Any]) -> str | None:
    for c in (
        item.get("advisory_url"),
        item.get("vendor_url"),
        item.get("cisa_url"),
        item.get("kev_url"),
        item.get("url"),
        item.get("link"),
    ):
        if not c:
            continue
        m = URL_FINDER_RE.findall(html.unescape(str(c)))
        if m:
            return m[0].strip("'")
    return None

def _remove_number_noise(s: str) -> str:
    # keep CVEs; drop long version strings and big numeric blobs & numeric-only parens
    s = VERSION_RE.sub("", s)
    s = LONG_DIGITS_RE.sub("", s)
    s = PAREN_NUM_RE.sub("", s)
    # tidy vendor phrasing
    s = re.sub(r"\b(is|are)\s+being\s+rolled\s+out\b", "is rolling out", s, flags=re.I)
    s = re.sub(r"\bincluding:\s*", "Includes ", s, flags=re.I)
    s = re.sub(r"\s+,", ",", s)
    s = WS_RE.sub(" ", s)
    return s.strip()

# ---- light jargon → plain-English ----------
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

# ------------ Summary builder -------------
def _summary_text(title: str, summary: str, *, tone: str) -> str:
    """
    Build a plain-English summary. For 'simple' tone: up to ~2 sentences.
    For 'detailed': up to 4 short sentences. Cleans number noise & jargon.
    """
    raw = _strip_html(summary or title or "")
    raw = _remove_number_noise(raw)
    raw = _plain_english(raw)

    parts = [p.strip(" .;:-") for p in SENT_SPLIT_RE.split(raw) if p.strip()]
    if not parts:
        return title.strip() + "."

    max_sentences = 2 if (tone or "simple") == "simple" else 4
    chosen = parts[:max_sentences]
    text = ". ".join(chosen).strip(". ") + "."
    # keep it readable, not too long
    hard_cap = 420 if (tone or "simple") == "simple" else 800
    if len(text) > hard_cap:
        text = text[: hard_cap - 1].rstrip() + "…"
    return text

# ------------ badges & actions -------------
def _matches(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def _build_actions(title: str, summary: str, tone: str) -> Tuple[List[str], List[str]]:
    t = f"{title} {summary}".lower()
    fix: List[str] = []
    verify: List[str] = []

    if _matches(t, "chrome", "chromium", "chromeos"):
        fix.append("Update Chrome/ChromeOS to the latest stable build.")
        if tone == "detailed":
            fix += ["Admins can force updates via policy.", "Restart browser/devices if needed."]
            verify += ["chrome://version shows the latest build on sample endpoints."]
        return fix, verify

    if _matches(t, "apple", "ios", "ipados", "macos", "safari"):
        fix.append("Update iPhone/iPad/Mac to the latest version.")
        if tone == "detailed":
            fix += ["If managed, push the OS update via MDM and enforce a restart."]
            verify += ["MDM inventory shows the minimum OS version across devices."]
        return fix, verify

    if _matches(t, "microsoft", "windows", "edge", "office", "exchange", "teams"):
        fix.append("Run Windows Update or deploy the latest security updates.")
        if tone == "detailed":
            fix += ["WSUS/Intune: approve and force install; reboot if required."]
            verify += ["MDM/VA shows the target KBs installed across the fleet."]
        return fix, verify

    # default
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
        sev = "HIGH" if kev or epss >= 0.70 else "MEDIUM"

    lines = [
        f"Security alert: {sev.title()}",
        f"Known Exploited Vulnerability: {'Yes' if kev else 'No'}",
    ]
    if epss >= 0.01:
        lines.append(f"EPSS: {epss:.2f}")
    return lines

# ------------ renderer -------------
def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    title = _tidy_title((item.get("title") or "").strip())
    meta  = _meta_lines(item)

    summary_src = item.get("summary") or item.get("content") or ""
    summary = _summary_text(title, summary_src, tone=(tone or "simple"))

    fix, verify = _build_actions(title, summary, tone or "simple")
    actions_lines = [f"{BULLET} {a}" for a in fix]

    doc_str = (item.get("_docs") or "").strip()

    blocks: List[str] = [
        f"*{idx}) {title}*",
        *meta,
        "",
        f"Summary: {summary}",
        "",
        "Fix:",
        *actions_lines,
    ]

    if (tone or "simple") == "detailed" and verify:
        blocks += ["", "Verify:", *[f"{BULLET} {v}" for v in verify]]

    if doc_str:
        blocks += ["", f"Docs: {doc_str}"]

    text = "\n".join(blocks).strip()
    return text[:2900] + "…" if len(text) > 2900 else text
