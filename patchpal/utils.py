# patchpal/utils.py
from __future__ import annotations
import re, html
from typing import Dict, Any, List, Tuple

# ---- HTML cleanup -----------------------------------------------------------
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
    """Best-effort HTML→text with entity unescape and bullet normalization."""
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

def _one_line(s: str, limit: int) -> str:
    """Collapse to one tight line for TL;DR."""
    s = _strip_html(s)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[: limit - 1].rstrip() + "…") if len(s) > limit else s

# ---- Doc picking ------------------------------------------------------------
def _pick_doc_url(item: Dict[str, Any]) -> str | None:
    # If selector injected multiple links as a preformatted string, use that.
    pre = item.get("_docs")
    if isinstance(pre, str) and pre.strip():
        return pre

    # Fallback: first valid http(s) across common fields
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
            return f"<{m[0]}|Vendor advisory / guidance>"
    return None

# ---- Badges -----------------------------------------------------------------
def _sev_badge(item: Dict[str, Any]) -> str:
    sev = (item.get("severity") or "").upper()
    kev = bool(item.get("kev") or item.get("known_exploited"))
    epss = item.get("epss")
    # Derive severity if not provided
    if not sev:
        try:
            e = float(epss or 0.0)
        except Exception:
            e = 0.0
        sev = "HIGH" if kev or e >= 0.70 else "MEDIUM"

    bits: List[str] = []
    emoji = ":rotating_light:" if sev in {"CRITICAL", "HIGH"} else ":warning:"
    bits.append(f"{emoji} *{sev}*")
    if kev:
        bits.append("• KEV")
    try:
        e = float(epss or 0.0)
        if e >= 0.01:                # hide meaningless zeros
            bits.append(f"• EPSS {e:.2f}")
    except Exception:
        pass
    return " ".join(bits)

# ---- Vendor-aware actions ---------------------------------------------------
def _matches(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def _actions_and_verify(title: str, summary: str, tone: str) -> Tuple[List[str], List[str]]:
    """Return (fix, verify) with vendor-first precedence to avoid false Windows hits."""
    t = f"{title} {summary}".lower()
    fix: List[str] = []
    verify: List[str] = []

    # 1) Browser / ChromeOS
    if _matches(t, "chrome", "chromium", "chromeos"):
        fix.append("Update Chrome/ChromeOS to the latest stable build.")
        if tone == "detailed":
            fix += ["Admin: enforce AutoUpdate / policy push.", "Restart browser/devices if needed."]
            verify += ["chrome://version (sample endpoints) shows the new build."]
        return fix, verify

    # 2) Apple platforms
    if _matches(t, "apple", "ios", "ipados", "macos", "safari"):
        fix.append("Update iOS/iPadOS/macOS to the latest version.")
        if tone == "detailed":
            fix += ["MDM: push update & enforce restart."]
            verify += ["MDM inventory reflects minimum OS version across fleet."]
        return fix, verify

    # 3) Citrix (avoid Windows false positives)
    if _matches(t, "citrix"):
        fix.append("Apply the Citrix vendor hotfix/update to affected servers.")
        if tone == "detailed":
            fix += ["Schedule maintenance; stop service, apply hotfix, restart, validate."]
            verify += ["Citrix build/version matches bulletin; VA scan is clean."]
        return fix, verify

    # 4) Git
    if t.startswith("git ") or " git " in t:
        fix.append("Upgrade Git to the latest patched version.")
        if tone == "detailed":
            fix += ["Update package/installer on developer workstations & CI runners.",
                    "Restart shells/IDEs that embed Git."]
            verify += ["`git --version` shows the patched release on sample hosts."]
        return fix, verify

    # 5) VMware / Cisco / Fortinet
    if _matches(t, "vmware"):
        fix.append("Apply the VMware security advisory patch.")
        if tone == "detailed":
            fix += ["Follow vendor KB; snapshot/backup before patching; restart as required."]
            verify += ["Build number matches advisory; service health OK."]
        return fix, verify

    if _matches(t, "cisco"):
        fix.append("Apply the Cisco advisory fix to affected devices.")
        if tone == "detailed":
            fix += ["Upgrade image; reload device during a maintenance window."]
            verify += ["`show version` / PSIRT guidance confirms fixed release."]
        return fix, verify

    if _matches(t, "fortinet", "fortigate"):
        fix.append("Upgrade Fortinet devices to the fixed firmware.")
        if tone == "detailed":
            fix += ["Backup config; upgrade; validate policies & logs."]
            verify += ["Firmware shows fixed build; PSIRT advisory closed out."]
        return fix, verify

    # 6) Microsoft (after vendor-specific checks)
    if _matches(t, "microsoft", "windows", "exchange", "office", "teams", "edge"):
        fix.append("Run Windows Update / deploy latest cumulative security updates.")
        if tone == "detailed":
            fix += ["WSUS/Intune: approve & force install; reboot if required."]
            verify += ["MDM/VA shows target KBs present across scope."]
        return fix, verify

    # 7) Adobe
    if _matches(t, "adobe", "acrobat", "reader"):
        fix.append("Update Adobe Acrobat/Reader to the latest release.")
        if tone == "detailed":
            fix += ["Push via software distribution; restart the app to complete."]
            verify += ["Help → About shows patched version on sample hosts."]
        return fix, verify

    # Default fallback
    fix.append("Apply the vendor security update/hotfix.")
    if tone == "detailed":
        fix += ["Schedule downtime if needed; restart & validate services."]
        verify += ["App/service version matches advisory; VA re-scan is clean."]
    return fix, verify

# ---- Public renderer --------------------------------------------------------
def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    """
    Slack mrkdwn with a practical ops voice.

    simple   -> TL;DR (short) + Fix (2 bullets) + Docs (if available)
    detailed -> TL;DR (longer) + Fix (step-by-step) + Verify + Docs
    """
    title = (item.get("title") or "").strip()
    badges = _sev_badge(item)

    raw_summary = item.get("summary") or item.get("content") or ""
    summary = _one_line(raw_summary, 200 if tone == "simple" else 380)

    fix, verify = _actions_and_verify(title, summary, tone)

    # Compose
    lines: List[str] = [f"*{idx}) {title}*", badges, f"*TL;DR:* {summary}"]
    lines.append(f"*Fix{' (step-by-step)' if tone == 'detailed' else ''}:*")
    lines += [f"• {a}" for a in fix]

    if tone == "detailed" and verify:
        lines.append("*Verify:*")
        lines += [f"• {v}" for v in verify]

    docs = _pick_doc_url(item)
    if docs:
        # If _docs was provided (preformatted), _pick_doc_url already returns the full string
        if docs.startswith("<http"):
            lines.append(f"*Docs:* {docs}")
        else:
            lines.append(f"*Docs:* {docs}")

    text = "\n".join([l for l in lines if l]).strip()
    # Slack section text cap ~3000 chars; keep a margin
    if len(text) > 2900:
        text = text[:2900] + "…"
    return text
