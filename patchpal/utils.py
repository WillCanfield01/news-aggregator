# patchpal/utils.py
from __future__ import annotations
import re, html
from typing import Dict, Any, List

# ---------- HTML -> Slack-safe text -----------------------------------------

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\f\v]+")
_NL = re.compile(r"\n{3,}")

def _strip_tags(raw: str) -> str:
    if not raw:
        return ""
    s = raw

    # normalize common blocky tags to newlines
    s = re.sub(r"(?i)</?(p|div|h[1-6]|section|article|blockquote|ul|ol)>", "\n", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)

    # convert <li>…</li> to bullets
    def _li_to_bullet(m):
        inner = _TAG.sub("", m.group(1))
        return f"\n• {inner.strip()}"

    s = re.sub(r"(?is)<li[^>]*>(.*?)</li>", _li_to_bullet, s)

    # drop script/style
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", s)

    # kill everything else, then unescape
    s = _TAG.sub("", s)
    s = html.unescape(s)

    # compact whitespace
    s = s.replace("\r", "")
    s = _WS.sub(" ", s)
    s = _NL.sub("\n\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = s.strip()
    return s


def _shorten(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0]
    return cut + "…"


# ---------- Risk badges ------------------------------------------------------

def _sev_badge(sev: str | None) -> str:
    s = (sev or "").lower()
    if s.startswith("crit"):
        return "🛑 CRITICAL"
    if s.startswith("high"):
        return "🚨 HIGH"
    if s.startswith("med"):
        return "⚠️ MEDIUM"
    if s.startswith("low"):
        return "ℹ️ LOW"
    return "⚠️"

def _kev_epss_badges(item: Dict[str, Any]) -> str:
    bits = []
    if item.get("kev") or item.get("known_exploited"):
        bits.append("KEV")
    try:
        epss = float(item.get("epss") or 0)
        if epss >= 0.85:
            bits.append(f"EPSS {epss:.2f} (99th pct)")
        elif epss >= 0.5:
            bits.append(f"EPSS {epss:.2f}")
    except Exception:
        pass
    return " • ".join(bits)


# ---------- Vendor-aware "Do this now" ---------------------------------------

def _vendor_from(item: Dict[str, Any]) -> str:
    v = (item.get("vendor_guess") or "").lower()
    title = (item.get("title") or "").lower()
    txt = f"{v} {title}"
    for key in ("microsoft", "windows", "office", "exchange", "edge", "teams"):
        if key in txt:
            return "microsoft"
    if "apple" in txt or "ios" in txt or "macos" in txt:
        return "apple"
    if "chrome" in txt or "google" in txt or "android" in txt:
        return "google"
    if "adobe" in txt:
        return "adobe"
    if "citrix" in txt:
        return "citrix"
    if "trend micro" in txt or "apex one" in txt:
        return "trend"
    if "vmware" in txt:
        return "vmware"
    if "cisco" in txt:
        return "cisco"
    return (v or "").split()[0] if v else "generic"

def _do_now_steps(item: Dict[str, Any], limit: int) -> List[str]:
    vendor = _vendor_from(item)
    product = item.get("product") or item.get("product_guess") or ""
    prod = product.strip() or (item.get("title") or "").split("—")[-1].strip()

    # sensible defaults per vendor
    if vendor == "microsoft":
        return [
            "Install the latest Windows/Office cumulative security updates.",
            "If you manage patches, approve & deploy via WSUS/Intune; reboot if required.",
            "Prioritize internet-facing and high-risk assets within 24–48 hours.",
        ][:limit]
    if vendor == "apple":
        return [
            "Update iOS/iPadOS/macOS to the latest stable version.",
            "If supervised, push the update via MDM; enforce AutoUpdate and schedule restarts.",
        ][:limit]
    if vendor == "google":
        return [
            "Update Chrome/Android to the latest stable version.",
            "For managed fleets, force relaunch after update; disable outdated versions.",
        ][:limit]
    if vendor == "citrix":
        return [
            f"Apply the vendor security update/hotfix for {prod} on all affected servers.",
            "Schedule maintenance: stop services, apply hotfix, restart services, validate.",
            "Check external exposure; prioritize gateway/edge systems within 24–48 hours.",
        ][:limit]
    if vendor == "trend":
        return [
            f"Upgrade/patch Trend Micro {prod} to the fixed build.",
            "Restrict external access and audit admin accounts until patched.",
        ][:limit]
    if vendor == "vmware":
        return [
            f"Apply the VMware update for {prod}; snapshot, patch, then validate.",
            "If internet-exposed, front-door with ACL/VPN until patched.",
        ][:limit]
    if vendor == "cisco":
        return [
            f"Upgrade Cisco {prod} to the fixed release per the advisory.",
            "If exposed, restrict management interfaces until patched.",
        ][:limit]

    # generic fallback
    return [
        "Apply the vendor-supplied security update to fixed versions.",
        "Prioritize internet-exposed and critical systems; reboot/validate after patch.",
    ][:limit]


def _docs_links(item: Dict[str, Any]) -> str:
    links = []
    # support multiple possible fields
    for key in ("advisory_url", "vendor_url", "cve_url", "details_url"):
        u = item.get(key)
        if u:
            links.append(("Advisory", u))
    docs = item.get("docs") or item.get("links") or []
    if isinstance(docs, dict):
        for k, v in docs.items():
            if v:
                links.append((k.title(), v))
    elif isinstance(docs, list):
        for v in docs:
            if isinstance(v, str):
                links.append(("Details", v))
            elif isinstance(v, dict) and v.get("url"):
                links.append((v.get("name") or "Details", v["url"]))
    if not links:
        return ""
    # de-dup by URL
    seen = set()
    out = []
    for name, url in links:
        if url in seen:
            continue
        seen.add(url)
        out.append(f"<{url}|{name}>")
        if len(out) >= 3:
            break
    return " · ".join(out)


# ---------- Public: render Slack text ---------------------------------------

def render_item_text_core(item: Dict[str, Any], idx: int, tone: str = "simple") -> str:
    title = item.get("title") or "Security Update"
    severity = _sev_badge(item.get("severity"))
    riskbits = _kev_epss_badges(item)
    badges = f"{severity}" + (f" • {riskbits}" if riskbits else "")

    # sanitize & choose summary length by tone
    raw_summary = item.get("summary") or item.get("content") or ""
    summary = _strip_tags(str(raw_summary))
    if tone == "simple":
        summary = _shorten(summary, 180)

    # actions
    bullet_cap = 2 if tone == "simple" else 5
    steps = _do_now_steps(item, bullet_cap)
    steps_txt = "\n".join([f"• {s}" for s in steps])

    # docs
    docs = _docs_links(item)
    see_more = f"\n_See more:_ {docs}" if docs and tone != "simple" else (f"\n_See more:_ {docs}" if docs and tone == "simple" else "")

    # build text
    header = f"*{idx}) {title}*\n{badges}"
    body = f"\n*What happened:* {summary}\n\n*Do this now{' (step-by-step)' if tone!='simple' else ''}:*\n{steps_txt}"
    return f"{header}{body}{see_more}".strip()
