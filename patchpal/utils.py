# patchpal/utils.py
from __future__ import annotations
import os, re, html
from typing import Dict, Any, List, Tuple

# --- Tunables (keep consistent with selector) -------------------------------
EPSS_THRESHOLD = float(os.getenv("PATCHPAL_EPSS_THRESHOLD", "0.70"))

# --- Sanitizers --------------------------------------------------------------
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

def _text_blob(*parts: str, limit: int | None = None) -> str:
    s = " ".join(p for p in parts if p).strip()
    s = _strip_html(s)
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s

def _matches(text: str, *needles: str) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def _extract_cves(*chunks: str) -> List[str]:
    found = []
    seen = set()
    for ch in chunks:
        if not ch:
            continue
        for c in CVE_RE.findall(ch):
            c = c.upper()
            if c not in seen:
                seen.add(c); found.append(c)
    return found

# --- Badges -----------------------------------------------------------------
def _sev_badge(item: Dict[str, Any]) -> str:
    sev = (item.get("severity") or "").upper()
    kev = bool(item.get("kev") or item.get("known_exploited"))
    # Accept both str/float; hide EPSS when 0.00
    try:
        epss = float(item.get("epss") or 0.0)
    except Exception:
        epss = 0.0

    if not sev:
        sev = "HIGH" if kev or epss >= EPSS_THRESHOLD else "MEDIUM"

    emoji = ":rotating_light:" if sev in {"CRITICAL", "HIGH"} else ":warning:"
    bits: List[str] = [f"{emoji} *{sev}*"]
    if kev:
        bits.append("KEV")
    if epss >= 0.01:
        bits.append(f"EPSS {epss:.2f}")
    return " • ".join(bits)

# --- Docs (1–3 practical links) --------------------------------------------
def _all_urls_from_item(item: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    # Scan common fields first
    fields = [
        "advisory_url","vendor_url","cisa_url","kev_url","msrc_url",
        "doc_url","source","url","link",
        "summary","content",
    ]
    for f in fields:
        v = item.get(f)
        if not v:
            continue
        if isinstance(v, list):
            for x in v:
                urls += URL_FINDER_RE.findall(html.unescape(str(x)))
        else:
            urls += URL_FINDER_RE.findall(html.unescape(str(v)))
    # Unique while preserving order
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _docs_links(item: Dict[str, Any]) -> str:
    links: list[tuple[str, str]] = []
    t = _strip_html(" ".join([str(item.get("title") or ""), str(item.get("summary") or ""), str(item.get("content") or "")])).lower()
    cves = _extract_cves(item.get("title",""), item.get("summary",""))

    def add(url: str | None, label: str):
        if not url:
            return
        links.append((url, label))

    # Microsoft
    if any(k in t for k in ("microsoft","windows","edge","office","exchange","teams","msrc")):
        if cves:
            add(f"https://msrc.microsoft.com/update-guide/vulnerability/{cves[0]}", "MSRC advisory")
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

    # Git
    if (" git " in (" " + t)) or (str(item.get("title","")).lower().startswith("git ")):
        add("https://git-scm.com/downloads", "Git downloads")
        add("https://github.com/git/git/tree/master/Documentation/RelNotes", "Git release notes")

    # Citrix
    if "citrix" in t:
        add("https://support.citrix.com/security-bulletins", "Citrix security bulletins")

    # Cisco / VMware / Fortinet generic portals
    if "cisco" in t:
        add("https://www.cisco.com/c/en/us/support/docs/psirt.html", "Cisco PSIRT")
    if "vmware" in t:
        add("https://www.vmware.com/security/advisories.html", "VMware advisories")
    if "fortinet" in t or "fortigate" in t:
        add("https://www.fortiguard.com/psirt", "Fortinet PSIRT")

    # Always include the vendor/source link if present
    all_urls = _all_urls_from_item(item)
    if all_urls:
        add(all_urls[0], "Vendor notice")

    # Dedup + cap to 3, format as Slack links
    out, seen = [], set()
    for u, label in links:
        if not u or u in seen:
            continue
        seen.add(u); out.append(f"<{u}|{label}>")
        if len(out) >= 3:
            break
    return " · ".join(out)

# --- Actions / Verify (Chrome → Apple → Microsoft → default) ----------------
def _actions_and_verify(title: str, summary: str, tone: str) -> Tuple[List[str], List[str]]:
    t = f"{title} {summary}".lower()
    fix: List[str] = []
    verify: List[str] = []

    # Chrome first (so Chrome blog posts mentioning 'Windows' don't fall into MS)
    if "chrome" in t:
        fix.append("Update Chrome to the latest stable.")
        if tone == "detailed":
            fix += ["Admin: force update via policy.", "Restart browser/devices if needed."]
            verify += ["chrome://version on sample endpoints shows latest build."]
        return fix, verify

    # Apple
    if any(k in t for k in ("apple","ios","ipados","macos","safari")):
        fix.append("Update iOS/iPadOS/macOS to the latest version.")
        if tone == "detailed":
            fix += ["MDM: push update and enforce restart."]
            verify += ["MDM inventory shows minimum OS version across devices."]
        return fix, verify

    # Microsoft
    if any(k in t for k in ("microsoft","windows","edge","office","exchange","teams")):
        fix.append("Run Windows Update / deploy latest security updates.")
        if tone == "detailed":
            fix += ["WSUS/Intune: approve & force install; reboot if required."]
            verify += ["MDM/VA shows target KBs installed on all scoped devices."]
        return fix, verify

    # Default
    fix.append("Apply the vendor security update/hotfix.")
    if tone == "detailed":
        fix += ["Schedule maintenance; restart if needed."]
        verify += ["App/service version matches advisory; VA re-scan is clean."]
    return fix, verify

# --- Public formatter --------------------------------------------------------
def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    """
    Build Slack mrkdwn in an 'ops voice':
      - Title
      - Badges (🚨/⚠️, KEV, EPSS if > 0)
      - TL;DR (one-liner)
      - Fix (bullets; detailed adds ops steps)
      - Verify (only in detailed)
      - Docs (1–3 links)
    """
    tone = (tone or "simple").lower()

    title = (item.get("title") or f"Item {idx}").strip()
    badges = _sev_badge(item)

    summary_src = item.get("summary") or item.get("content") or ""
    summary = _text_blob(summary_src, limit=220 if tone == "simple" else 420)

    fix, verify = _actions_and_verify(title, summary, tone)
    docs = _docs_links(item)

    lines: List[str] = [
        f"*{idx}) {title}*",
        badges,
        f"*TL;DR:* {summary}" if summary else "",
        f"*Fix{' (step-by-step)' if tone == 'detailed' else ''}:*",
        *[f"• {a}" for a in fix],
    ]
    if tone == "detailed" and verify:
        lines += ["*Verify:*", *[f"• {v}" for v in verify]]
    if docs:
        lines.append(f"*Docs:* {docs}")

    text = "\n".join([ln for ln in lines if ln]).strip()
    # Slack section hard limit ~3k
    if len(text) > 2900:
        text = text[:2900] + "…"
    return text
