import requests, feedparser
from datetime import datetime, timedelta
from dateutil import parser as dtp

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
VENDOR_FEEDS = [
    ("Microsoft MSRC", "https://msrc.microsoft.com/blog/rss"),
    ("Cisco PSIRT", "https://sec.cloudapps.cisco.com/security/center/rss.x?i=44"),
]

def fetch_cisa_kev(days=14, limit=50):
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
        items.append({
            "title": f"{v.get('cveID')} — {v.get('vendorProject','Unknown')} {v.get('product','')}",
            "summary": v.get("shortDescription") or v.get("vulnerabilityName") or "",
            "link": v.get("notes") or "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
            "source": "CISA KEV",
            "what_to_do": "Patch/mitigate now; prioritize internet-exposed assets and verify with a scan.",
        })
    return items[:limit]

def fetch_vendor_advisories(limit=20):
    items = []
    for name, url in VENDOR_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:limit]:
                items.append({
                    "title": e.get("title"),
                    "summary": e.get("summary") or e.get("description") or "",
                    "link": e.get("link"),
                    "source": name,
                    "what_to_do": "Apply vendor’s fixed version; audit exposure; compensating controls if delay.",
                })
        except Exception:
            continue
    return items

KEYWORDS = ["zero-day", "actively exploited", "out-of-band", "RCE", "authentication bypass", "CVE-"]

def score_item(it: dict) -> int:
    t = (it.get("title") or "") + " " + (it.get("summary") or "")
    score = 0
    score += sum(1 for k in KEYWORDS if k.lower() in t.lower()) * 3
    if it.get("source") == "CISA KEV":
        score += 8
    if "CVE-" in t:
        score += 2
    return score

def top3_today():
    pool = fetch_cisa_kev(days=14, limit=80) + fetch_vendor_advisories(limit=20)
    pool.sort(key=score_item, reverse=True)
    seen = set()
    result = []
    for it in pool:
        ttl = (it.get("title") or "").strip()
        if ttl and ttl not in seen:
            seen.add(ttl)
            result.append(it)
        if len(result) == 3:
            break
    return result

def as_slack_blocks(items):
    blocks = [
        {"type":"header","text":{"type":"plain_text","text":"Today’s Top 3 Patches / CVEs","emoji":True}}
    ]
    for i, it in enumerate(items, 1):
        text = f"*{i}) {it['title']}*\n{it['summary']}\n\n*What to do:* {it['what_to_do']}\n<{it['link']}|Advisory / Details>"
        blocks.append({"type":"section","text":{"type":"mrkdwn","text":text}})
        if i < len(items):
            blocks.append({"type":"divider"})
    blocks.append({"type":"context","elements":[{"type":"mrkdwn","text":"Auto-posted by PatchPal • Sources: CISA KEV + vendor advisories"}]})
    return blocks
