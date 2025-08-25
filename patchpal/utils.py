# patchpal/utils.py
from __future__ import annotations
import re, html
from typing import Dict, Any, List

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
    # Grab the first valid http(s) in any candidate field
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

def _sev_badge(item: Dict[str, Any]) -> str:
    sev = (item.get("severity") or "").upper()
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = item.get("epss")
    bits: List[str] = []
    if sev in {"CRITICAL", "HIGH", "MEDIUM"}:
        emoji = ":rotating_light:" if sev in {"CRITICAL", "HIGH"} else ":warning:"
        bits.append(f"{emoji} {sev}")
    if kev:
        bits.append("• KEV")
    try:
        e = float(epss or 0)
        if e > 0:
            bits.append(f"• EPSS {e:.2f}")
    except Exception:
        pass
    return " ".join(bits) if bits else ""

def _text_blob(*parts: str, limit: int | None = None) -> str:
    s = " ".join(p for p in parts if p).strip()
    s = _strip_html(s)
    if limit and len(s) > limit:
        s = s[:limit - 1].rstrip() + "…"
    return s

def _matches(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def _build_actions(title: str, summary: str, tone: str) -> List[str]:
    t = f"{title} {summary}".lower()
    # Very light routing by vendor/stack to make actions feel specific.
    if _matches(t, "windows", "microsoft", "edge"):
        if tone == "detailed":
            return [
                "Apply latest cumulative/security updates on servers and endpoints.",
                "If you manage patches: WSUS → approve & deploy; Intune → set deadline and force restart.",
                "Reboot if required; re-scan/validate.",
                "Prioritize internet-exposed/critical assets within 24–48 hours.",
            ]
        return [
            "Run Windows Update / deploy latest security updates.",
            "WSUS/Intune: approve & force install; reboot if required.",
        ]

    if _matches(t, "apple", "ios", "macos", "ipados"):
        if tone == "detailed":
            return [
                "Update iPhone/iPad/Mac to the latest available versions.",
                "If managed, push the OS update via MDM and enforce restart.",
                "Re-scan/validate after update.",
            ]
        return ["Update iOS/iPadOS/macOS to the latest version.", "Push via MDM; enforce restart if needed."]

    if _matches(t, "chrome", "google chrome"):
        if tone == "detailed":
            return [
                "Update Chrome/Chromium to latest stable.",
                "Enforce AutoUpdate in policy; relaunch to complete.",
            ]
        return ["Update Chrome to latest stable.", "Ensure AutoUpdate is on; relaunch browser."]

    if _matches(t, "adobe", "acrobat", "reader"):
        if tone == "detailed":
            return [
                "Update Adobe Acrobat/Reader to the latest release.",
                "If managed, push via your software distribution tool; restart apps to apply.",
            ]
        return ["Update Adobe Acrobat/Reader to latest build.", "Restart the app to apply patches."]

    # Generic vendor advisory fallback
    if tone == "detailed":
        return [
            "Apply the vendor-supplied security update/hotfix to affected systems.",
            "Schedule maintenance as needed; stop services, apply update, restart, validate.",
        ]
    return ["Apply vendor security update.", "Validate and re-scan after patching."]

def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    """
    Build a Slack-mrkdwn string for a single item.
    `tone='simple'` => short summary + 2 bullets.
    `tone='detailed'` => longer summary + step-by-step + doc link.
    """
    title = (item.get("title") or "").strip()
    badges = _sev_badge(item)
    summary_src = item.get("summary") or item.get("content") or ""
    summary = _text_blob(summary_src, limit=400 if tone == "simple" else 700)

    header = f"*{idx}) {title}*\n{badges}".strip()
    what = f"*What happened:* {summary}" if summary else ""

    actions = _build_actions(title, summary, tone)
    if tone == "detailed":
        actions_title = "*Do this now (step-by-step):*"
    else:
        actions_title = "*Do this now:*"

    actions_text = "\n".join([f"• {a}" for a in actions])

    out = [header]
    if what:
        out.append(what)
    out.append(actions_title)
    out.append(actions_text)

    if tone == "detailed":
        doc = _pick_doc_url(item)
        if doc:
            out.append(f"*Docs:* <{doc}|Vendor advisory / guidance>")

    return "\n".join(out).strip()
