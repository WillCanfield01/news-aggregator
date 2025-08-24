# patchpal/selector.py
import re
import requests, feedparser
from datetime import datetime, timedelta
from dateutil import parser as dtp

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
EPSS_API = "https://api.first.org/data/v1/epss"  # ?cve=CVE-...,CVE-...
NVD_API  = "https://services.nvd.nist.gov/rest/json/cves/2.0"  # ?cveId=CVE-...

URL_RE = re.compile(r"https?://[^\s>)\"']+")
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)

def first_url(s: str | None) -> str | None:
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
    r"\bKEV\b": "CISA KEV (known exploited)",
}

def plainify(s: str | None) -> str:
    if not s:
        return ""
    out = s
    for pat, rep in ACRONYM_MAP.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = out.replace("an attacker could", "attackers could").replace("may allow", "could allow")
    return out

def vendor_action_steps(title: str, source: str) -> list[str]:
    tl = (title or "").lower()
    src = (source or "").lower()
    if any(k in tl for k in ["apple", "ios", "macos", "ipados"]) or "apple" in src:
        return [
            "Open *Settings → General → Software Update* (macOS: *System Settings → General → Software Update*).",
            "Install all available security updates.",
            "Restart the device(s) after installing.",
        ]
    if any(k in tl for k in ["microsoft", "windows", "exchange", "sql server"]) or "msrc" in src:
        return [
            "Run *Windows Update* (or WSUS/Intune) and install the latest security updates.",
            "Reboot servers/workstations to complete the update.",
            "If servers need a window, schedule within 24–48 hours.",
        ]
    if "cisco" in tl or "cisco" in src:
        return [
            "Open the Cisco advisory; locate the *fixed release* for your product.",
            "Upgrade to the recommended fixed version.",
            "If you must delay, apply Cisco’s mitigations and restrict management access.",
        ]
    if any(k in tl for k in ["fortinet", "fortios", "fortigate"]):
        return [
            "Upgrade to the Fortinet fixed build listed in the advisory.",
            "Disable exposed services until patched and restrict management access.",
            "Review admin/auth logs for unusual activity.",
        ]
    if any(k in tl for k in ["vmware", "esxi", "vcenter"]):
        return [
            "Apply the VMware patch for your exact build.",
            "Block external access to management interfaces until patched.",
            "Monitor for suspicious logins.",
        ]
    return [
        "Apply the vendor’s *fixed version* as soon as possible (today if internet-facing).",
        "If you must delay, apply published mitigations and restrict external access.",
        "After patching, run a scan and review admin logs.",
    ]

def kev_links(cve_id: str) -> list[tuple[str, str]]:
    return [
        ("NVD", f"https://nvd.nist.gov/vuln/detail/{cve_id}"),
        ("CISA KEV", f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog?search={cve_id}"),
        ("CVE.org", f"https://www.cve.org/CVERecord?id={cve_id}"),
    ]

def link_text(links: list[tuple[str, str]]) -> str:
    return " · ".join(f"<{u}|{l}>" for l, u in links if u)

# ------------------ feeds ------------------

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
        summary = plainify(v.get("shortDescription") or v.get("vulnerabilityName") or "")
        items.append({
            "title": title,
            "summary": summary,
            "source": "CISA KEV",
            "cve": cve,
            "links": kev_links(cve),
            "steps": vendor_action_steps(title, "CISA KEV"),
            "is_kev": True,
        })
    return items[:limit]

VENDOR_FEEDS = [
    ("Microsoft MSRC", "https://msrc.microsoft.com/blog/rss"),
    ("Cisco PSIRT", "https://sec.cloudapps.cisco.com/security/center/rss.x?i=44"),
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
                m = CVE_RE.search(" ".join([title, summary]))
                cve = m.group(0).upper() if m else None
                base = {
                    "title": title,
                    "summary": summary,
                    "source": name,
                    "cve": cve,
                    "links": [("Advisory", link)] if link else [],
                    "steps": vendor_action_steps(title, name),
                    "is_kev": False,
                }
                if cve:
                    base["links"] += kev_links(cve)  # add stable fallbacks
                items.append(base)
        except Exception:
            continue
    return items

# ------------------ scoring ------------------

KEYWORDS_URGENT = ["zero-day", "actively exploited", "authentication bypass", "remote code execution", "RCE", "CVE-"]

def score_item(it: dict) -> float:
    t = ((it.get("title") or "") + " " + (it.get("summary") or "")).lower()
    score = 0.0
    score += sum(1 for k in KEYWORDS_URGENT if k.lower() in t) * 3
    if it.get("is_kev"):
        score += 10
    if it.get("cve"):
        score += 2
    # EPSS boost (added after enrichment; default 0)
    if it.get("epss"):
        score += min(it["epss"] * 10, 10)  # up to +10
    return score

# ------------------ enrichment ------------------

def fetch_epss_map(cves: list[str]) -> dict[str, dict]:
    if not cves:
        return {}
    try:
        resp = requests.get(EPSS_API, params={"cve": ",".join(cves)}, timeout=15)
        data = resp.json().get("data", [])
        out = {}
        for row in data:
            try:
                out[row["cve"].upper()] = {
                    "epss": float(row.get("epss", 0.0)),
                    "percentile": float(row.get("percentile", 0.0)),
                }
            except Exception:
                continue
        return out
    except Exception:
        return {}

def fetch_cvss(cve: str) -> tuple[str | None, float | None]:
    """Return (severity, baseScore) from NVD for one CVE."""
    try:
        r = requests.get(NVD_API, params={"cveId": cve}, timeout=15).json()
        vuln = (r.get("vulnerabilities") or [{}])[0].get("cve", {})
        metrics = vuln.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            arr = metrics.get(key)
            if arr:
                data = arr[0].get("cvssData", {})
                sev = data.get("baseSeverity") or arr[0].get("baseSeverity")
                score = data.get("baseScore") or arr[0].get("baseScore")
                return (sev, float(score) if score is not None else None)
    except Exception:
        pass
    return (None, None)

def enrich(items: list[dict]) -> list[dict]:
    # EPSS in batch
    cves = [it["cve"] for it in items if it.get("cve")]
    epss_map = fetch_epss_map(cves)
    for it in items:
        cve = it.get("cve")
        if cve and cve.upper() in epss_map:
            it["epss"] = epss_map[cve.upper()]["epss"]
            it["epss_pct"] = epss_map[cve.upper()]["percentile"]
        # CVSS per-item (only a few items)
        if cve:
            sev, score = fetch_cvss(cve)
            it["cvss_severity"] = sev
            it["cvss_score"] = score
    return items

def severity_badge(it: dict) -> str:
    badges = []
    sev = (it.get("cvss_severity") or "").upper()
    score = it.get("cvss_score")
    if sev:
        emoji = ":rotating_light:" if sev == "CRITICAL" else (":warning:" if sev == "HIGH" else (":large_yellow_square:" if sev == "MEDIUM" else ":white_small_square:"))
        if score is not None:
            badges.append(f"{emoji} *{sev}* {score:.1f}")
        else:
            badges.append(f"{emoji} *{sev}*")
    if it.get("epss") is not None:
        pct = int(round((it.get("epss_pct") or 0.0) * 100))
        badges.append(f"EPSS *{it['epss']:.2f}* ({pct}th pct)")
    if it.get("is_kev"):
        badges.append("*KEV*")
    return " · ".join(badges)

# ------------------ selection ------------------

def top3_today():
    pool = fetch_cisa_kev(days=14, limit=80) + fetch_vendor_advisories(limit=25)
    # Enrich pool *lightly*: only keep EPSs/CVSS for items we might show
    # First, coarse sort to take the top ~20, then enrich and final sort
    pool.sort(key=lambda x: ((x.get("is_kev") and 1) or 0, "CVE-" in (x.get("title") or "")), reverse=True)
    shortlist = enrich(pool[:20])
    shortlist.sort(key=score_item, reverse=True)

    seen = set()
    result = []
    for it in shortlist:
        ttl = (it.get("title") or "").strip()
        if not ttl or ttl in seen:
            continue
        seen.add(ttl)
        result.append(it)
        if len(result) == 3:
            break
    return result

# ------------------ Slack formatting ------------------

def as_slack_blocks(items, tone: str = "simple"):
    blocks = [{"type": "header", "text": {"type": "plain_text", "text": "Today’s Top 3 Patches / CVEs", "emoji": True}}]
    for i, it in enumerate(items, 1):
        title = it.get("title") or ""
        summary = it.get("summary") or ""
        steps = it.get("steps") or []
        links_line = link_text(it.get("links", []))
        badges = severity_badge(it)

        if tone == "simple":
            # very plain: 1–2 sentence what happened, then 2–3 bullets
            text = f"*{i}) {title}*\n"
            if badges:
                text += f"{badges}\n"
            text += f"*What happened:* {summary}\n\n*Do this now:*\n"
            for s in steps[:3]:
                text += f"• {s}\n"
        else:
            # detailed: keep everything + links
            text = f"*{i}) {title}*\n"
            if badges:
                text += f"{badges}\n"
            text += f"*What happened:* {summary}\n\n*Do this now:*\n"
            for s in steps:
                text += f"• {s}\n"
            if links_line:
                text += f"\n{links_line}"

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
        if i < len(items):
            blocks.append({"type": "divider"})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal · by The Real Roundup"}]})
    return blocks


# patchpal/selector.py  (only changed/added pieces)

def topN_today(n: int = 3):
    """Return the top N items after enrichment + scoring."""
    n = max(1, min(10, int(n)))  # safety
    pool = fetch_cisa_kev(days=14, limit=80) + fetch_vendor_advisories(limit=50)

    # quick pre-sort to keep enrichment cheap, then enrich a shortlist
    pool.sort(key=lambda x: (x.get("is_kev", False), bool(x.get("cve"))), reverse=True)
    shortlist = enrich(pool[: max(20, n * 5)])
    shortlist.sort(key=score_item, reverse=True)

    seen = set()
    out = []
    for it in shortlist:
        ttl = (it.get("title") or "").strip()
        if not ttl or ttl in seen:
            continue
        seen.add(ttl)
        out.append(it)
        if len(out) == n:
            break
    return out

def as_slack_blocks(items, tone: str = "simple"):
    count = len(items) or 0
    blocks = [{
        "type": "header",
        "text": {"type": "plain_text", "text": f"Today’s Top {count} Patches / CVEs", "emoji": True}
    }]
    # ... (rest of your block formatting unchanged)
