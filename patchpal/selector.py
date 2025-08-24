# patchpal/selector.py
import re
import requests, feedparser
from datetime import datetime, timedelta
from dateutil import parser as dtp

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# --- Helpers ---------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s>)\"']+")

def first_url(s: str | None) -> str | None:
    """Return the first http(s) URL from a messy string, or None."""
    if not s:
        return None
    m = URL_RE.search(s)
    return m.group(0) if m else None

ACRONYM_MAP = {
    r"\bRCE\b": "remote code execution",
    r"\bPoC\b": "proof of concept",
    r"\bDoS\b": "denial of service",
    r"\bXSS\b": "cross-site scripting",
    r"\bLPE\b": "local privilege escalation",
    r"\bpriv(ilege)? esc(alation)?\b": "privilege escalation",
    r"\b0[- ]day\b": "zero-day",
    r"\bKEV\b": "CISA KEV (known exploited)",
}

def plainify(s: str | None) -> str:
    """Small, opinionated simplifier for human-friendly text."""
    if not s:
        return ""
    out = s
    for pat, rep in ACRONYM_MAP.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    # soften vendor boilerplate
    out = out.replace("an attacker could", "attackers could")
    out = out.replace("may allow", "could allow")
    return out

def vendor_action_hint(title: str, source: str) -> str:
    tl = (title or "").lower()
    src = (source or "").lower()
    if any(k in tl for k in ["apple", "ios", "macos", "ipados"]) or "apple" in src:
        return ("Apple devices: open *Settings → General → Software Update* (macOS: *System Settings → General → "
                "Software Update*) and install all available security updates. Restart devices after installing.")
    if any(k in tl for k in ["microsoft", "windows", "exchange", "sql server"]) or "msrc" in src:
        return ("Run *Windows Update* (or WSUS/Intune) and install the latest security updates. Reboot. "
                "For servers, schedule a maintenance window within 24–48 hours.")
    if "cisco" in tl or "cisco" in src:
        return ("Check the Cisco advisory for your product and install the *fixed release*. "
                "If you can’t patch today, apply Cisco’s *temporary mitigations* and restrict access with ACLs.")
    if any(k in tl for k in ["fortinet", "fortios", "fortigate"]):
        return ("Upgrade to the Fortinet *fixed build* listed in the advisory. If upgrade is delayed, "
                "disable exposed services and limit management access to a secure network.")
    if any(k in tl for k in ["trend micro"]):
        return ("Update the Trend Micro product to the fixed version in the advisory and restrict external access "
                "until patched. Review recent admin logins for anomalies.")
    if any(k in tl for k in ["vmware", "esxi", "vcenter"]):
        return ("Apply the VMware patch for your exact build. If patching is delayed, block external access "
                "to management interfaces and monitor for suspicious logins.")
    # Default
    return ("Apply the vendor’s *fixed version* as soon as possible. If the system is internet-facing, patch *today*. "
            "Otherwise patch within 72 hours. After patching, run a scan and review logs for unusual admin activity.")

def kev_links(cve_id: str) -> list[tuple[str, str]]:
    return [
        ("NVD", f"https://nvd.nist.gov/vuln/detail/{cve_id}"),
        ("CISA KEV", f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog?search={cve_id}"),
        ("CVE.org", f"https://www.cve.org/CVERecord?id={cve_id}")
    ]

def link_text(links: list[tuple[str, str]]) -> str:
    parts = []
    for label, url in links:
        if url:
            parts.append(f"<{url}|{label}>")
    return " · ".join(parts)

# --- Feeds -----------------------------------------------------------------

def fetch_cisa_kev(days=14, limit=80):
    try:
        data = requests.get(KEV_URL, timeout=20).json()
    except Exception:
        return []
    cutoff = datetime.utcnow() - timedelta(days=days)
    items = []
    for v in data.get("vulnerabilities", []):
        try:
            added = dtp.parse(v.get("dateAdded")).replace(tzinfo=None)
        except Exception:
            continue
        if added < cutoff:
            continue
        cve = v.get("cveID")
        vendor = v.get("vendorProject", "Unknown")
        product = v.get("product", "")
        title = f"{cve} — {vendor} {product}".strip()
        summary = v.get("shortDescription") or v.get("vulnerabilityName") or ""
        # Clean, friendly text
        summary = plainify(summary)
        items.append({
            "title": title,
            "summary": summary,
            "source": "CISA KEV",
            "links": kev_links(cve),
            "what_to_do": vendor_action_hint(title, "CISA KEV") + " This item is in *CISA KEV*, meaning it’s being exploited in the wild.",
        })
    return items[:limit]

VENDOR_FEEDS = [
    ("Microsoft MSRC", "https://msrc.microsoft.com/blog/rss"),
    ("Cisco PSIRT", "https://sec.cloudapps.cisco.com/security/center/rss.x?i=44"),
    # add more as you grow
]

def fetch_vendor_advisories(limit=25):
    items = []
    for name, url in VENDOR_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:limit]:
                raw_link = e.get("link") or e.get("id") or e.get("summary") or ""
                link = first_url(raw_link)
                title = plainify(e.get("title") or "")
                summary = plainify(e.get("summary") or e.get("description") or "")
                items.append({
                    "title": title,
                    "summary": summary,
                    "source": name,
                    "links": [("Advisory", link)] if link else [],
                    "what_to_do": vendor_action_hint(title, name),
                })
        except Exception:
            continue
    return items

# --- Scoring / selection ---------------------------------------------------

KEYWORDS_URGENT = ["zero-day", "actively exploited", "out-of-band", "authentication bypass", "remote code execution", "RCE", "CVE-"]

def score_item(it: dict) -> int:
    t = ((it.get("title") or "") + " " + (it.get("summary") or "")).lower()
    score = 0
    score += sum(1 for k in KEYWORDS_URGENT if k.lower() in t) * 3
    if it.get("source") == "CISA KEV":
        score += 8
    if "cve-" in t:
        score += 2
    return score

def top3_today():
    pool = fetch_cisa_kev(days=14, limit=80) + fetch_vendor_advisories(limit=25)
    pool.sort(key=score_item, reverse=True)
    seen = set()
    result = []
    for it in pool:
        ttl = (it.get("title") or "").strip()
        if not ttl or ttl in seen:
            continue
        seen.add(ttl)
        result.append(it)
        if len(result) == 3:
            break
    return result

# --- Slack blocks ----------------------------------------------------------

def as_slack_blocks(items):
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "Today’s Top 3 Patches / CVEs", "emoji": True}}
    ]
    for i, it in enumerate(items, 1):
        title = it.get("title") or ""
        summary = it.get("summary") or ""
        action = it.get("what_to_do") or "Apply the vendor’s fixed version and verify with a scan."
        links_line = link_text(it.get("links", []))
        text = (
            f"*{i}) {title}*\n"
            f"*What happened:* {summary}\n\n"
            f"*Do this now:* {action}\n"
        )
        if links_line:
            text += f"\n{links_line}"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
        if i < len(items):
            blocks.append({"type": "divider"})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal • Sources: CISA KEV + vendor advisories"}]})
    return blocks
