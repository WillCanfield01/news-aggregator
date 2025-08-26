# patchpal/utils.py
from __future__ import annotations
import re, html
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
LONG_DIGITS_RE  = re.compile(r"\b\d{6,}\b")            # ids like 427681143
PAREN_NUM_RE    = re.compile(r"\((?:[^a-zA-Z]*\d[^)]*)\)")  # (Platform Version: 16093.115.0)
SENT_SPLIT_RE   = re.compile(r"(?<=[\.\?!])\s+")
WS_RE           = re.compile(r"\s+")

WORDS_RE        = re.compile(r"[A-Za-z0-9.+#/_-]+")
URL_FINDER_RE   = re.compile(r"https?://[^\s>]+")

# Visual bullet
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
    for c in (item.get("advisory_url"), item.get("vendor_url"), item.get("cisa_url"),
              item.get("kev_url"), item.get("url"), item.get("link")):
        if not c:
            continue
        m = URL_FINDER_RE.findall(html.unescape(str(c)))
        if m:
            return m[0]
    return None

def _remove_number_noise(s: str) -> str:
    # keep CVEs; drop long version strings and big numeric blobs & numeric-only parens
    s = VERSION_RE.sub("", s)
    s = LONG_DIGITS_RE.sub("", s)
    s = PAREN_NUM_RE.sub("", s)
    # tiny copyedits that often appear in vendor blogs
    s = re.sub(r"\b(is|are)\s+being\s+rolled\s+out\b", "is rolling out", s, flags=re.I)
    s = re.sub(r"\bincluding:\s*", "Includes ", s, flags=re.I)
    s = re.sub(r"\s+,", ",", s)
    s = WS_RE.sub(" ", s)
    return s.strip()

def _tldr(title: str, summary: str, max_chars: int = 220, max_sentences: int = 2) -> str:
    raw = _strip_html(summary or "") or (title or "")
    raw = _remove_number_noise(raw)
    parts = [p.strip(" .;:-") for p in SENT_SPLIT_RE.split(raw) if p.strip()]
    out, used = "", 0
    for p in parts:
        if used >= max_sentences:
            break
        candidate = (f"{out} {p}").strip() if out else p
        if len(candidate) > max_chars:
            if not out:
                # hard cut first sentence if it's too long
                return (p[: max_chars - 1] + "…")
            break
        out = candidate
        used += 1
    out = out.rstrip(",;: ")
    if not out.endswith((".", "!", "?")):
        out += "."
    return out

# ------------ badges & actions -------------
def _sev_badge(item: Dict[str, Any]) -> str:
    sev = (item.get("severity") or "").upper()
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = item.get("epss")
    bits: List[str] = []
    # derive severity if not provided
    if not sev:
        try:
            e = float(epss or 0)
        except Exception:
            e = 0.0
        sev = "HIGH" if kev or e >= 0.70 else "MEDIUM"
    emoji = ":rotating_light:" if sev in {"CRITICAL","HIGH"} else ":warning:"
    bits.append(f"{emoji} *{sev}*")
    if kev:
        bits.append("KEV")
    try:
        e = float(epss or 0)
        if e >= 0.01:  # hide zero
            bits.append(f"EPSS {e:.2f}")
    except Exception:
        pass
    return " • ".join(bits)

def _matches(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def _build_actions(title: str, summary: str, tone: str) -> Tuple[List[str], List[str]]:
    t = f"{title} {summary}".lower()
    fix: List[str] = []
    verify: List[str] = []

    if _matches(t, "chrome", "chromium"):
        fix.append("Update Chrome/ChromeOS to the latest stable build.")
        if tone == "detailed":
            fix += ["Admin: force update via policy.", "Restart browser/devices if needed."]
            verify += ["chrome://version shows latest build on sample endpoints."]
        return fix, verify

    if _matches(t, "apple", "ios", "ipados", "macos", "safari"):
        fix.append("Update iOS/iPadOS/macOS to the latest version.")
        if tone == "detailed":
            fix += ["MDM: push update and enforce restart."]
            verify += ["MDM inventory shows minimum OS version across devices."]
        return fix, verify

    if _matches(t, "microsoft", "windows", "edge", "office", "exchange", "teams"):
        fix.append("Run Windows Update / deploy latest security updates.")
        if tone == "detailed":
            fix += ["WSUS/Intune: approve & force install; reboot if required."]
            verify += ["MDM/VA shows target KBs installed fleetwide."]
        return fix, verify

    # default
    fix.append("Apply the vendor security update/hotfix.")
    if tone == "detailed":
        fix += ["Schedule maintenance; restart if needed."]
        verify += ["Service/app version matches advisory; VA re-scan is clean."]
    return fix, verify

# ------------ renderer -------------
def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    title = (item.get("title") or "").strip()
    badges = _sev_badge(item)
    summary_src = item.get("summary") or item.get("content") or ""

    # sentence-aware, number-clean TL;DR
    max_chars = 360 if tone == "simple" else 700
    summary = _tldr(title, summary_src, max_chars=max_chars, max_sentences=2)

    header = f"*{idx}) {title}*"
    if badges:
        header += f"\n{badges}"

    fix, verify = _build_actions(title, summary, tone)

    lines: List[str] = [
        header,
        f"*TL;DR:* {summary}",
        "",
        f"*Fix{' (step-by-step)' if tone == 'detailed' else ''}:*",
        *[f"{BULLET} {a}" for a in fix],
    ]

    if tone == "detailed" and verify:
        lines += ["", "*Verify:*", *[f"{BULLET} {v}" for v in verify]]

    doc_str = (item.get("_docs") or "").strip()
    if doc_str:
        lines += ["", f"*Docs:* {doc_str}"]

    text = "\n".join(lines).strip()
    return text[:2900] + "…" if len(text) > 2900 else text
