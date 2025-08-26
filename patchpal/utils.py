# patchpal/utils.py
from __future__ import annotations
import re, html
from typing import Dict, Any, List

# --- HTML & text cleanup -----------------------------------------------------
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

# long naked digit blobs (keep versions/CVEs; strip tracker IDs like 427681143)
LONG_DIGITS_RE  = re.compile(r"\b\d{7,}\b")
# lop off vendor blog boilerplate if present
BOILERPLATE_CUT = re.compile(
    r"(Interested in switching release channels\?|Find out how\.|"
    r"If you find a new issue.*|community help forum.*|Read more.*)$",
    re.I,
)

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

def _smart_trim(sent: str, limit: int) -> str:
    """Trim to a sentence/clause boundary near the limit; fall back to hard cut."""
    s = re.sub(r"\s+", " ", (sent or "").strip())
    if not s or len(s) <= limit:
        return s
    # clip to limit, then try to end on punctuation
    clip = s[:limit]
    # try period/question/exclamation if it appears after 60% of limit
    cut = max(clip.rfind("."), clip.rfind("?"), clip.rfind("!"))
    if cut >= int(0.6 * limit):
        return clip[:cut + 1]
    # otherwise try a clause boundary
    for ch in (";", "—", "–", "-"):
        pos = clip.rfind(ch)
        if pos >= int(0.6 * limit):
            return clip[:pos].rstrip()
    # final fallback
    return clip.rstrip() + "…"

def _tidy_summary(src: str, limit: int) -> str:
    """Clean vendor text, strip tracker IDs/boilerplate, then smart-trim."""
    s = _strip_html(src)
    # remove long naked numbers (leave versions/CVEs alone)
    s = LONG_DIGITS_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" .-—")
    # lop boilerplate if present
    m = BOILERPLATE_CUT.search(s)
    if m:
        s = s[: m.start()].rstrip()
    return _smart_trim(s, limit)

# --- Doc picking -------------------------------------------------------------
def _pick_doc_url(item: Dict[str, Any]) -> str | None:
    # first valid http(s) in any candidate field
    candidates = [
        item.get("advisory_url"),
        item.get("vendor_url"),
        item.get("cisa_url"),
        item.get("kev_url"),
        item.get("url"),
        item.get("link"),
    ]
    for c in candidates:
        if not c:
            continue
        m = URL_FINDER_RE.findall(html.unescape(str(c)))
        if m:
            return m[0]
    return None

# --- Badges ------------------------------------------------------------------
def _sev_badge(item: Dict[str, Any]) -> str:
    sev = (item.get("severity") or "").upper()
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = item.get("epss")
    bits: List[str] = []
    if sev in {"CRITICAL", "HIGH", "MEDIUM"}:
        emoji = ":rotating_light:" if sev in {"CRITICAL", "HIGH"} else ":warning:"
        bits.append(f"{emoji} *{sev}*")
    if kev:
        bits.append("KEV")
    try:
        e = float(epss or 0)
        if e >= 0.01:  # hide meaningless zero
            bits.append(f"EPSS {e:.2f}")
    except Exception:
        pass
    return " • ".join(bits) if bits else ""

# --- Targeted actions --------------------------------------------------------
def _matches(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def _build_actions(title: str, summary: str, tone: str) -> List[str]:
    t = f"{title} {summary}".lower()

    # Chrome first (Windows mention in blogs can otherwise hijack)
    if _matches(t, "chrome", "chromium"):
        return ([
            "Update Chrome/ChromeOS to the latest stable."
        ] + ([
            "Admin: force update via policy; relaunch to complete.",
            "Verify chrome://version on sample endpoints."
        ] if tone == "detailed" else []))

    # Apple
    if _matches(t, "apple", "ios", "ipados", "macos", "safari"):
        return ([
            "Update iOS/iPadOS/macOS to the latest version."
        ] + ([
            "MDM: push update and enforce restart.",
            "Verify MDM inventory meets minimum OS version."
        ] if tone == "detailed" else []))

    # Microsoft / Windows
    if _matches(t, "windows", "microsoft", "edge", "msrc", "office", "exchange", "teams"):
        return ([
            "Run Windows Update / deploy the latest security updates."
        ] + ([
            "WSUS/Intune: approve & force install; reboot if required.",
            "Validate with VA/MDM that target KBs are installed."
        ] if tone == "detailed" else []))

    # Adobe
    if _matches(t, "adobe", "acrobat", "reader"):
        return ([
            "Update Adobe Acrobat/Reader to the latest release."
        ] + ([
            "Push via your software distribution tool; restart the app to apply.",
        ] if tone == "detailed" else []))

    # Generic fallback
    return ([
        "Apply the vendor security update/hotfix."
    ] + ([
        "Schedule maintenance; restart services if needed; re-scan to confirm."
    ] if tone == "detailed" else []))

# --- Main renderer -----------------------------------------------------------
def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    """
    Build Slack mrkdwn with a clear 'ops voice':
      • TL;DR (clean sentence)
      • Fix (2–3 bullets; detailed adds ops/verify)
      • Docs (up to 3 links, passed in via item['_docs'] by selector)
    """
    tone = (tone or "simple").lower()

    title   = (item.get("title") or f"Item {idx}").strip()
    badges  = _sev_badge(item)

    # Cleaned, sentence-aware TL;DR
    src = item.get("summary") or item.get("content") or ""
    limit = 260 if tone == "simple" else 520
    tldr = _tidy_summary(src, limit)

    actions = _build_actions(title, tldr, tone)
    docs_str = (item.get("_docs") or "")  # selector should set this; otherwise we’ll fall back
    if not docs_str:
        doc = _pick_doc_url(item)
        if doc:
            docs_str = f"<{doc}|Vendor notice>"

    lines: List[str] = []
    lines.append(f"*{idx}) {title}*")
    if badges:
        lines.append(badges)
    if tldr:
        lines.append(f"*TL;DR:* {tldr}")
    if actions:
        lines.append(f"*Fix{' (step-by-step)' if tone == 'detailed' else ''}:*")
        lines += [f"• {a}" for a in actions]
    if docs_str:
        lines.append(f"*Docs:* {docs_str}")

    out = "\n".join(lines).strip()
    return out if len(out) <= 2900 else out[:2900] + "…"
