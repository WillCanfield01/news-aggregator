import os
import re
import json
import random
import difflib
import requests
from datetime import datetime, date, timedelta
import openai
from flask import Blueprint, render_template, jsonify
from markdown2 import markdown
from unidecode import unidecode
from app.models import CommunityArticle
from app.extensions import db
from rapidfuzz import fuzz
import feedparser
from urllib.parse import urlparse
import hashlib
from app.subscriptions import current_user_is_plus
import json

# ------------------------------
# Config & constants
# ------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # unused for cyber; safe to keep
ADD_PERSONAL_NOTES = False
RECENT_WINDOW = 10  # uniqueness window for titles
VARIETY_PENALTY = 40  # kept (not used in cyber scorer but harmless)

openai.api_key = OPENAI_API_KEY

bp = Blueprint("all_articles", __name__, url_prefix="/all-articles")

ARTICLES_DIR = os.path.join(os.path.dirname(__file__), "generated_articles")
os.makedirs(ARTICLES_DIR, exist_ok=True)

USED_ITEMS_FILE = os.path.join(ARTICLES_DIR, "_used_items.json")
# Dedupe thresholds (tunable)
TOPIC_OVERLAP_BLOCK = 0.58        # block if same-topic overlap >= 0.58
TITLE_SIMILARITY_BLOCK = 82       # block if title similarity > 82
VENDOR_DAY_CAP = 2                # max items per vendor per day

# Relaxed pass if we didn't reach n items
RELAXED_TOPIC_OVERLAP = 0.70
RELAXED_TITLE_SIMILARITY = 86

def _load_used_items():
    try:
        with open(USED_ITEMS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_used_items(data: dict):
    try:
        with open(USED_ITEMS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def _item_key(it: dict) -> str:
    # stable key per article item; prefer link, fall back to title
    raw = (_normalize(it.get("link") or "") + "|" + _normalize(it.get("title") or "")).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ------------------------------
# Preview helpers
# ------------------------------
def _strip_markdown(md_text: str) -> str:
    """
    Minimal markdown-to-plain helper for preview generation.
    Drops images, bullet lines, and photo credits so they don't leak into previews.
    """
    t = md_text or ""
    lines = []
    for ln in t.splitlines():
        raw = ln.strip()
        if not raw:
            continue
        if re.match(r"!\[[^\]]*\]\([^\)]+\)", raw):
            continue
        if re.match(r"[-*+]\s+", raw):
            continue
        if re.search(r"photo by", raw, re.I):
            continue
        lines.append(raw)
    t = "\n".join(lines)
    t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[*_`]", "", t)
    return t.strip()


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    """Return the first N sentences from text."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences]).strip()


def _sentence_limited_summary(text: str, max_chars: int = 330) -> str:
    """
    Keep complete sentences up to max_chars; append ellipsis if truncated.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    out: list[str] = []
    total = 0
    for s in sentences:
        if not s:
            continue
        proposed = total + len(s) + (1 if out else 0)
        if out and proposed > max_chars:
            break
        if not out and len(s) > max_chars:
            out.append(s[: max_chars].rstrip(" .,!?:;") + "…")
            return out[0]
        out.append(s)
        total += len(s) + (1 if out else 0)
    joined = " ".join(out).strip()
    truncated = len(joined) < len(text.strip())
    if truncated and not joined.endswith("…"):
        joined = joined.rstrip(".!? ") + "…"
    return joined


def _extract_bullets_from_md(md_text: str, max_items: int = 4) -> list[str]:
    bullets: list[str] = []
    for line in (md_text or "").splitlines():
        m = re.match(r"^\s*[-*+]\s+(.*)", line)
        if m:
            val = m.group(1).strip()
            if val:
                bullets.append(val)
        if len(bullets) >= max_items:
            break
    return bullets[:max_items]


def _fallback_bullets_from_text(text: str, max_items: int = 4) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    out = []
    for s in sentences[1:]:
        s = s.strip()
        if len(s) < 8:
            continue
        out.append(s)
        if len(out) >= max_items:
            break
    return out or sentences[:max_items]


def _coerce_preview_bullets(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(b).strip() for b in raw if str(b).strip()]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return _coerce_preview_bullets(parsed)
        except Exception:
            pass
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        return parts
    return []


def _extract_hero_info(md_text: str) -> tuple[str | None, str | None]:
    url = None
    credit = None
    img = re.search(r"!\[[^\]]*\]\((https?://[^\s\)]+)\)", md_text or "")
    if img:
        url = img.group(1)
    credit_match = re.search(r"photo by[^\\n]*", md_text or "", re.I)
    if credit_match:
        credit = credit_match.group(0)
        credit = credit.strip("* ").strip()
    return url, credit


def _build_preview(article, md_text: str) -> tuple[str, list[str]]:
    summary = getattr(article, "preview_summary", None) or article.meta_description or ""
    plain = _strip_markdown(md_text)
    if not summary:
        summary = _first_sentences(plain, 3)
    summary = re.sub(r"photo by[^\\n]*", "", summary, flags=re.I)
    summary = re.sub(r"[-*•]\s+", "", summary)
    clean_summary = _sentence_limited_summary(summary, max_chars=340)
    return clean_summary.strip(), []

# Topic/vendor helpers
VENDOR_HINTS = [
    "apple","microsoft","google","cisco","vmware","citrix","fortinet","adobe","juniper",
    "dell","hp","intel","amd","sap","oracle","apache","nginx","okta","github","gitlab",
    "mozilla","zoom","slack","cloudflare","1password","lastpass"
]
GROUP_HINTS = ["lockbit","alphv","blackcat","clop","lazarus","apt29","apt28","sandworm",
               "fin7","scattered spider","black basta"]

STOPWORDS_TOPIC = {
    "the","a","an","and","or","to","of","in","on","for","with","as","by","from","is","are",
    "this","that","today","new","update","patch","patched","patches","security","vulnerability",
    "vulnerabilities","zero-day","exploit","researchers","attackers","ransomware","breach"
}

def _extract_cves(text: str):
    return re.findall(r'cve-\d{4}-\d+', text.lower())

def _detect_vendor(text: str):
    low = text.lower()
    for v in VENDOR_HINTS:
        if v in low:
            return v
    return None

def _topic_signature(it: dict) -> str:
    """Stable per-topic fingerprint for day-wide dedupe."""
    t = _normalize(f"{it.get('title','')} {it.get('summary','')}")
    low = t.lower()
    cves = _extract_cves(low)[:2]
    vendor = _detect_vendor(low)
    groups = [g for g in GROUP_HINTS if g in low][:2]
    # rare tokens = first few non-stopword 4+ char tokens
    tokens = [w for w in re.findall(r"[a-z0-9\-/\.]+", low)
              if len(w) >= 4 and w not in STOPWORDS_TOPIC][:6]
    parts = []
    if cves:   parts.append(",".join(cves))
    if vendor: parts.append(vendor)
    if groups: parts.append(",".join(groups))
    if tokens: parts.append(",".join(tokens))
    return "|".join(parts)

def _topic_overlap(sig_a: str, sig_b: str) -> float:
    """Jaccard overlap of signature parts."""
    A = set(re.split(r"[|,]", sig_a)) if sig_a else set()
    B = set(re.split(r"[|,]", sig_b)) if sig_b else set()
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# --- NEW: vetted cybersecurity RSS/Atom sources ---
CYBER_FEEDS = [
    "https://feeds.feedburner.com/TheHackersNews",                         # The Hacker News
    "https://www.bleepingcomputer.com/feed/",                              # BleepingComputer
    "https://krebsonsecurity.com/feed/",                                   # Krebs
    "https://www.securityweek.com/feed/",                                  # SecurityWeek
    "https://www.darkreading.com/rss.xml",                                 # DarkReading
    "https://www.cisa.gov/news-events/cybersecurity-advisories/alerts.xml",# CISA Alerts
    "https://msrc.microsoft.com/blog/feed",                                # Microsoft MSRC
    "https://googleprojectzero.blogspot.com/feeds/posts/default?alt=rss",  # Project Zero
]

# Keywords to score/cluster daily context
CYBER_KEYWORDS = [
    "cve", "zero-day", "zeroday", "ransomware", "data breach", "breach", "exploit",
    "patch", "vulnerability", "vulnerabilities", "supply chain", "apt", "phishing",
    "rce", "privilege escalation", "poc", "proof-of-concept", "ioc", "indicators of compromise",
    "cisa", "nsa", "fbi", "mitre", "atlas", "lockbit", "blackcat", "alphv"
]

AI_KEYWORDS = [
    "ai", "gpt", "chatgpt", "llm", "automation", "autogen",
    "midjourney", "stability", "image generator", "prompt",
    "agent", "agents", "rag", "langchain", "openai", "ollama"
]

BANNED_WORDS = [
    "sex", "sexual", "nsfw", "porn", "nude", "nudes", "vagina", "penis", "erection",
    "boobs", "boob", "breast", "cum", "orgasm", "masturbat", "anal", "ass", "butt",
    "dick", "cock", "blowjob", "suck", "f***", "shit", "piss", "rape", "molest",
    "incest", "adult", "fetish", "taboo", "explicit", "onlyfans"
]
EXTRA_BANNED = ["hitler", "nazi", "terror", "rape", "porn", "sex", "nsfw"]

GPTISMS = [
    "delve", "dive into", "navigate", "vibrant", "comprehensive", "pivotal", "notably",
    "realm", "landscape", "tapestry", "embark", "unpack", "delving", "navigating", "explore",
    "in today's world", "in today's society", "at the end of the day", "from all walks of life",
    "it's worth noting", "in conclusion", "ultimately", "pivot", "essentially", "moreover",
]

GENERIC_PHRASES = [
    "how to make money with ai",
    "ways to make money",
    "side hustle",
    "earn money",
    "build confidence",
    "general tips"
]

AI_TOOL_KEYWORDS = [
    "chatgpt", "midjourney", "dall-e", "stability ai", "elevenlabs", 
    "runwayml", "firefly", "gpt", "stable diffusion", "claude",
    "notion ai", "jasper", "copy ai", "perplexity", "elicit", "research rabbit"
]

ACTION_KEYWORDS = [
    "sell", "create", "generate", "design", "automate", 
    "write", "build", "develop", "publish", "train", "launch"
]

IT_BANNED_TOPICS = [
    "identity", "workflow", "authentication", "sso", "iam",
    "infrastructure", "firewall", "endpoint", "it management",
    "ticketing", "active directory"
]

MONETIZATION_KEYWORDS = [
    "etsy", "fiverr", "upwork", "youtube", "tiktok", "shopify", 
    "airbnb", "gumroad", "amazon", "kdp", "patreon", "substack"
]

# Who we write for (helps generate distinct angles)
AUDIENCE_KEYWORDS = {
    "teachers": ["teacher", "classroom", "lesson", "worksheet", "students", "school"],
    "students": ["student", "college", "campus", "dorm", "exam"],
    "parents": ["parent", "stay-at-home", "mom", "dad", "family", "kids"],
    "retirees": ["retiree", "retirement", "senior"],
    "freelancers": ["freelancer", "client", "gig", "portfolio"],
    "local businesses": ["local business", "restaurant", "salon", "gym", "realtor", "plumber"],
    "creators": ["youtuber", "tiktok", "podcast", "channel", "creator", "blog"],
    "etsy sellers": ["etsy", "printables", "stickers", "svg", "planner"],
}

DELIVERABLE_KEYWORDS = [
    "printables","journal","journals","planner","planners","worksheet","worksheets",
    "wall art","poster","posters","stickers","svg","templates","template",
    "thumbnails","voiceover","voiceovers","e-cards","ecards","e-card","cover","covers",
    "intro","outro","scripts","script","prompts","prompt","logo","logos","mockups",
    "pack","bundle","bundles"
]

# Monetization “models” (broader than marketplace keywords)
MONETIZATION_MODELS = {
    "content": ["youtube", "tiktok", "shorts", "channel", "podcast", "blog", "newsletter", "substack", "patreon"],
    "services": ["fiverr", "upwork", "client", "gig", "freelance", "agency"],
    "products": ["etsy", "gumroad", "shopify", "kdp", "amazon", "print on demand", "print-on-demand"],
    "affiliate": ["affiliate", "review site", "roundup", "comparison", "seo"],
    "data/research": ["dataset", "research", "summary", "lead list", "prospect list"],
}

BUZZWORD_PENALTIES = [
    "easy money", "quick cash", "get rich", "make money fast",
    "passive income overnight", "no work", "in minutes", "side hustle hack"
]
NICHE_HINTS = [
    "etsy", "youtube", "shorts", "tiktok", "fiverr", "upwork",
    "plumber", "realtor", "teacher", "wedding", "podcast", "printables",
    "voiceover", "local business", "newsletter"
]
TOOL_HINTS = [
    "chatgpt", "gpt", "midjourney", "elevenlabs", "stable diffusion",
    "perplexity", "notion ai", "zapier", "make.com", "canva"
]

def _score_reel_variant(text: str) -> float:
    """Heuristic score: prefer concrete, honest, curiosity-driving openings."""
    t = text.lower()

    # 1) Hook length (first line should be ≤ 8 words)
    first_line = text.splitlines()[0].strip()
    hook_words = len(first_line.split())
    hook_score = 8 if 1 <= hook_words <= 8 else max(0, 12 - hook_words)

    # 2) Numbers & specifics (+$ amounts, %, steps, time)
    numerics = len(re.findall(r'(\$?\d+[%]?)', t))
    time_refs = len(re.findall(r'\b(\d+\s*(min|mins|minute|minutes|hour|hours|days))\b', t))
    steps_refs = len(re.findall(r'\b(step|steps|beat|beats)\b', t))
    specifics_score = numerics * 3 + time_refs * 2 + steps_refs

    # 3) Mentions of niche/tool terms
    niche_score = sum(1 for k in NICHE_HINTS if k in t)
    tool_score = sum(1 for k in TOOL_HINTS if k in t)

    # 4) Honesty bonus: “not instant”, “no magic”, “but here’s how”
    honesty_bonus = 2 if re.search(r"\b(not\s+instant|not\s+overnight|no\s+magic)\b", t) else 0

    # 5) Penalties for hype/buzzwords
    hype_pen = sum(2 for b in BUZZWORD_PENALTIES if b in t)

    # 6) Total length cap (keep it tight; 30–40s ~= 85–110 words spoken)
    words = len(re.findall(r'\w+', t))
    length_pen = 0 if 70 <= words <= 120 else abs(words - 95) * 0.5

    return hook_score + specifics_score + niche_score + tool_score + honesty_bonus - hype_pen - length_pen

# ------------------------------
# Helpers
# ------------------------------
def is_safe(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    for w in BANNED_WORDS:
        if w in t:
            return False
    return True

def detect_tool(text: str):
    low = text.lower()
    for k in AI_TOOL_KEYWORDS:
        if k in low:
            return k
    return None

def detect_action_kw(text: str):
    low = text.lower()
    for k in ACTION_KEYWORDS:
        if k in low:
            return k
    return None

def detect_platform(text: str):
    # finer than detect_monetization_method
    low = text.lower()
    for kw in MONETIZATION_KEYWORDS:
        if kw in low:
            return kw
    return None

def detect_model_bucket(text: str):
    low = text.lower()
    for model, words in MONETIZATION_MODELS.items():
        if any(w in low for w in words):
            return model
    return None

def detect_audience(text: str):
    low = text.lower()
    for audience, words in AUDIENCE_KEYWORDS.items():
        if any(w in low for w in words):
            return audience
    # light heuristics
    if "teacher" in low or "worksheet" in low: return "teachers"
    if "realtor" in low: return "local businesses"
    return None

def build_signature(tool, action, platform, audience):
    # compact signature for variety tracking
    return "|".join([tool or "-", action or "-", platform or "-", audience or "-"])

def recent_signatures(window=RECENT_WINDOW):
    recents = (CommunityArticle.query
               .order_by(CommunityArticle.date.desc(), CommunityArticle.id.desc())
               .limit(window)
               .all())
    sigs = []
    for a in recents:
        t = (a.title or "")
        tool = detect_tool(t)
        action = detect_action_kw(t)
        platform = detect_platform(t)
        audience = detect_audience(t)
        sigs.append(build_signature(tool, action, platform, audience))
    return set(sigs)

def uniqueness_penalty_for_title(title: str, window=RECENT_WINDOW):
    tool = detect_tool(title)
    action = detect_action_kw(title)
    platform = detect_platform(title)
    audience = detect_audience(title)
    sig = build_signature(tool, action, platform, audience)

    recents = recent_signatures(window)
    penalty = 0
    # Heavy hit if the exact combo appeared recently
    if sig in recents:
        penalty += 60
    # Lighter hits if individual pieces are overused
    if any(s.split("|")[0] == (tool or "-") for s in recents):      # tool seen
        penalty += 15
    if any(s.split("|")[1] == (action or "-") for s in recents):    # action seen
        penalty += 10
    if any(s.split("|")[2] == (platform or "-") for s in recents):  # platform seen
        penalty += 25
    if audience and any(s.split("|")[3] == audience for s in recents):
        penalty += 8
    return penalty

def detect_monetization_method(text):
    t = text.lower()
    for kw in MONETIZATION_KEYWORDS:
        if kw in t:
            return kw
    return None

def is_duplicate(title):
    existing = [a.title for a in CommunityArticle.query.all()]
    for old in existing:
        if fuzz.ratio(title.lower(), old.lower()) > 85:
            return True
    return False

def sanitize_gptisms(text: str) -> str:
    for phrase in GPTISMS:
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def is_generic(text):
    t = text.lower()
    return any(phrase in t for phrase in GENERIC_PHRASES)

def contains_tool_and_action(text):
    t = text.lower()
    has_tool = any(tool in t for tool in AI_TOOL_KEYWORDS)
    has_action = any(action in t for action in ACTION_KEYWORDS)
    has_deliverable = any(d in t for d in DELIVERABLE_KEYWORDS)
    has_it_banned = any(bad in t for bad in IT_BANNED_TOPICS)
    # accept tool + (action OR deliverable)
    return has_tool and (has_action or has_deliverable) and not has_it_banned

def is_monetizable_side_gig(title):
    t = title.lower()
    return contains_tool_and_action(t) and not any(bad in t for bad in IT_BANNED_TOPICS)

def remove_gpt_dashes(text: str) -> str:
    text = re.sub(r'(\s)—(\s)', r'\1,\2', text)
    text = re.sub(r'([a-zA-Z]),?\s*—\s*([a-zA-Z])', r'\1, \2', text)
    text = re.sub(r'—{2,}', '—', text)
    return text

def strip_unwanted_bold(text: str) -> str:
    def replacer(m):
        content = m.group(1)
        return "**Personal Note:**" if content.strip().lower() == "personal note:" else content
    return re.sub(r"\*\*(.*?)\*\*", replacer, text)

def clean_title(title: str) -> str:
    title = re.sub(r'\b([Rr]/)?AskReddit\b:?\s*', '', title or "")
    title = re.sub(r'\b[Rr]eddit\b:?\s*', '', title)
    return title.strip()

def split_markdown_sections(md: str):
    pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(md))
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(md)
        heading = match.group(2).strip()
        content = md[start:end].strip()
        sections.append((heading, content))
    return sections

def get_unsplash_image(query):
    if not UNSPLASH_ACCESS_KEY:
        return (None, None, None, None)
    url = "https://api.unsplash.com/photos/random"
    params = {"query": query, "orientation": "landscape", "client_id": UNSPLASH_ACCESS_KEY}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 200:
        data = r.json()
        alt = data.get("alt_description") or data.get("description") or ""
        return data["urls"]["regular"], data["user"]["name"], data["links"]["html"], alt
    return (None, None, None, None)

# ------------------------------
# Cybersecurity news ingest (RSS/Atom) → pick daily context
# ------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _domain(link: str) -> str:
    try:
        return urlparse(link).netloc.replace("www.", "")
    except Exception:
        return ""

def fetch_cyber_feed_entries(max_per_feed: int = 25) -> list:
    """Fetch recent entries from vetted cyber feeds."""
    items = []
    now = datetime.utcnow()
    for url in CYBER_FEEDS:
        parsed = feedparser.parse(url)
        for e in parsed.entries[:max_per_feed]:
            title = _normalize(getattr(e, "title", ""))
            summary = _normalize(getattr(e, "summary", "") or getattr(e, "description", ""))
            link = getattr(e, "link", "")
            # prefer published_parsed; fallback to now
            try:
                published = datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") else now
            except Exception:
                published = now
            if not title or not link:
                continue
            items.append({
                "title": title,
                "summary": summary,
                "link": link,
                "source": _domain(link),
                "published": published,
                "text": f"{title}\n\n{summary}"
            })
    return items

def score_cyber_item(item: dict) -> float:
    """Heuristic score combining recency & keyword density."""
    text = (item.get("title","") + " " + item.get("summary","")).lower()
    kw_hits = sum(1 for k in CYBER_KEYWORDS if k in text)
    # fresh = within ~72h is best
    hours_old = max(0, (datetime.utcnow() - item["published"]).total_seconds() / 3600.0)
    recency_score = max(0, 72 - hours_old)  # drop after 3 days
    # favor items from .gov/.edu or trusted brands a bit
    src = item.get("source", "")
    trust_bonus = 8 if (src.endswith(".gov") or src.endswith(".edu") or "cisa" in src or "msrc" in src) else 0
    return kw_hits * 12 + recency_score + trust_bonus

def select_unique_for_today(ranked_items: list, n: int = 9) -> list:
    """
    Pick up to n items NEW vs everything already used today:
    - blocks same-topic items via fingerprint overlap
    - avoids titles similar to earlier editions
    - soft-cap: max VENDOR_DAY_CAP items per vendor across the whole day
    - relaxed second pass if we didn't reach n
    """
    today = date.today().isoformat()
    used_map = _load_used_items()

    # ---- back-compat for old _used_items.json schema ----
    raw_day = used_map.get(today, {})
    if isinstance(raw_day, dict):
        used_keys = set(raw_day.get("keys", []))
        used_titles = list(raw_day.get("titles", []))
        used_topics = list(raw_day.get("topics", []))
        vendor_cnt = dict(raw_day.get("vendor_counts", {}))
    elif isinstance(raw_day, list):  # old schema: just a list of keys
        used_keys, used_titles, used_topics, vendor_cnt = set(raw_day), [], [], {}
    else:
        used_keys, used_titles, used_topics, vendor_cnt = set(), [], [], {}

    chosen = []

    # ---------- first pass: strict ----------
    for it in ranked_items:
        if len(chosen) >= n:
            break
        key = _item_key(it)
        title = (it.get("title") or "").strip()
        if not title or key in used_keys:
            continue

        # block similar to any title already used today
        if any(fuzz.ratio(title.lower(), t.lower()) > TITLE_SIMILARITY_BLOCK for t in used_titles):
            continue

        sig = _topic_signature(it)
        # block same-topic overlap
        if sig and any(_topic_overlap(sig, s) >= TOPIC_OVERLAP_BLOCK for s in used_topics):
            continue

        vendor = _detect_vendor(f"{it.get('title','')} {it.get('summary','')}")
        if vendor and vendor_cnt.get(vendor, 0) >= VENDOR_DAY_CAP:
            continue

        chosen.append(it)
        used_keys.add(key)
        used_titles.append(title)
        if sig:
            used_topics.append(sig)
        if vendor:
            vendor_cnt[vendor] = vendor_cnt.get(vendor, 0) + 1

    # ---------- second pass: relaxed (only if needed) ----------
    if len(chosen) < n:
        for it in ranked_items:
            if len(chosen) >= n:
                break
            key = _item_key(it)
            title = (it.get("title") or "").strip()
            if not title or key in used_keys:
                continue

            # relaxed title threshold
            if any(fuzz.ratio(title.lower(), t.lower()) > RELAXED_TITLE_SIMILARITY for t in used_titles):
                continue

            sig = _topic_signature(it)
            # relaxed topic overlap (allow a bit closer but not same)
            if sig and any(_topic_overlap(sig, s) >= RELAXED_TOPIC_OVERLAP for s in used_topics):
                continue

            # ignore vendor cap on relaxed pass
            chosen.append(it)
            used_keys.add(key)
            used_titles.append(title)
            if sig:
                used_topics.append(sig)

    # persist day state
    used_map[today] = {
        "keys": list(used_keys),
        "titles": used_titles,
        "topics": used_topics,
        "vendor_counts": vendor_cnt,
    }
    _save_used_items(used_map)
    return chosen

def pick_top_cyber_items(items: list, n: int = 8) -> list:
    dedup = {}
    for it in items:
        key = _normalize(it["title"]).lower()
        if key not in dedup:
            dedup[key] = it
        else:
            # keep earlier/better-scored one
            if score_cyber_item(it) > score_cyber_item(dedup[key]):
                dedup[key] = it
    ranked = sorted(dedup.values(), key=score_cyber_item, reverse=True)
    return ranked[:n]

# ------------------------------
# Generation prompts (Cyber Briefing)
# ------------------------------
def rewrite_title_for_cyber_briefing(items: list) -> str:
    """Create a clean briefing title WITHOUT the date (the template shows date below)."""
    top = items[:3]
    hints = []
    for it in top:
        t = (it["title"] or "").lower()
        if "cve-" in t and "CVE alerts" not in hints:
            hints.append("CVE alerts")
        if "ransomware" in t and "Ransomware" not in hints:
            hints.append("Ransomware")
        if ("zero-day" in t or "zero day" in t) and "Zero-day" not in hints:
            hints.append("Zero-day")
    suffix = f" — {', '.join(hints)}" if hints else ""
    return f"Daily Cybersecurity Briefing{suffix}"

def generate_accessible_brief(items: list, n_items: int = 9) -> str:
    """
    Write a skimmable daily brief for a general audience:
    - 7–10 bullets with 'What happened' + 'What to do'
    - No links, no citations, no dates in the title text
    - Plus 'If You Only Do 3 Things Today' and 'For Teams (super quick)'
    """
    # compact context for the model
    ctx_lines = []
    for it in items[:max(7, min(n_items, 10))]:
        title = (it.get("title") or "")[:180]
        summary = (it.get("summary") or "")[:220]
        ctx_lines.append(f"- {title} — {summary}")
    ctx = "\n".join(ctx_lines)

    prompt = f"""
You're writing Today's Quick Cyber Brief in plain English for a general audience.

Use ONLY the context list (titles + short summaries) to understand themes.
Do NOT include links, sources, domains, or citations. Don't add the date.

Write valid Markdown with exactly this structure:

## Today’s Quick Cyber Brief

- 7–10 bullet items. For EACH item, use this format:
  **Short headline**  
  *What happened:* one short sentence.  
  *What to do:* one short sentence.

## If You Only Do 3 Things Today
Provide a 2-column Markdown table:
| Action (1 minute each) | Why it matters |

## For Teams (super quick)
3–5 bullets with concise actions for IT/SecOps.

Keep it friendly, non-technical, and practical. No hype, no jargon walls.
Context:
{ctx}
"""
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1100,
        temperature=0.35
    )
    return resp.choices[0].message.content

def generate_reel_script_veed(article_text: str, topic: str) -> str:
    """
    Create 30–40s vertical script variants for the Daily Cyber Brief.
    Returns the best-scoring variant (fallback provided if API fails).
    """
    prompt = f"""
Create 5 short vertical video script VARIANTS for a Daily Cybersecurity Brief.
Each variant:
- The reel script should intruige viewers to watch all the way through, and also gives them the info they need to be secure
- 30–40 seconds (~90–110 words)
- First line [HOOK] ≤ 8 words, curiosity-based, no hype
- Short lines (≤ 12 words) for captions
- Name 3 concrete items if present (e.g., CVE id, vendor patch)
- Honest tone (no sensationalism)
- End with [CTA] “Full brief at TheRealRoundup.com”

Topic: {topic}

Reference (use only as context; don't quote verbatim):
{article_text[:1200]}

Return JSON exactly as: {{"variants": ["...","...","...","...","..."]}}
""".strip()

    try:
        resp = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        variants = [v.strip() for v in data.get("variants", []) if v.strip()]
        if not variants:
            raise ValueError("no variants")
        # pick the strongest using your scoring heuristic
        return max(variants, key=_score_reel_variant)
    except Exception:
        # safe fallback
        return (
            "Today’s biggest cyber risks, fast\n"
            "[B-ROLL: security dashboard]\n"
            "Zero-day patched in a major app\n"
            "Ransomware hits a regional hospital\n"
            "[B-ROLL: headlines montage]\n"
            "If you manage endpoints, patch today\n"
            "Rotate creds and block known C2 IPs\n"
            "[B-ROLL: terminal / firewall rules]\n"
            "[CTA] Full brief at TheRealRoundup.com"
        )


# ------------------------------
# Image suggestions (unchanged pattern)
# ------------------------------
def generate_section_image_suggestion(headline, keywords, outline, section_heading, section_content):
    prompt = (
        f"Given the article '{headline}' (keywords: {', '.join(keywords)}), "
        f"and the section below:\n\n"
        f"Section Heading: {section_heading}\n"
        f"Section Content:\n{section_content}\n\n"
        "Suggest a relevant image for Unsplash, with:\n"
        "- An image search query (max 5 words)\n"
        "- A short, descriptive caption and alt text (max 20 words)\n"
        "Format as JSON: {\"query\": \"...\", \"caption\": \"...\"}\n"
        "If no image fits, reply: {\"query\": \"\", \"caption\": \"skip\"}"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60, temperature=0.2, response_format={"type": "json_object"}
    )
    try:
        suggestion = json.loads(response.choices[0].message.content)
        if suggestion.get("caption", "").lower() == "skip" or not suggestion.get("query"):
            return None
        suggestion["section"] = section_heading
        return suggestion
    except Exception:
        return None

def split_first_faq(article_md: str):
    sections = split_markdown_sections(article_md)
    for h, content in sections:
        if h.strip().lower().startswith("faq"):
            return h, content
    return None

def humanize_reflection(text: str) -> str:
    # keep your quirky randomizer
    quirks = [
        lambda s: re.sub(r"\bthe\b", "teh", s, count=1),
        lambda s: re.sub(r"\b(I think|I guess|maybe|honestly)\b", r"\1, \1", s, count=1),
        lambda s: s.rstrip(".!?") + " I dunno.",
        lambda s: s.rstrip(".!?") + " Just saying.",
        lambda s: re.sub(r"\.$", "…", s, 1),
        lambda s: s[:random.randint(int(len(s)*0.6), int(len(s)*0.85))].rstrip(".!?") + "…"
    ]
    if random.random() < 0.40:
        text = random.choice(quirks)(text)
    return text

def fix_broken_personal_note(note: str) -> str:
    note = re.sub(r'\n{2,}', ' ', note)
    note = re.sub(r'\n?\*(\w+)\*\n?', '', note)
    note = re.sub(r'\n([a-zA-Z]{1,10})\n', ' ', note)
    note = re.sub(r'\n+', ' ', note).strip()
    sentences = re.split(r'(?<=[.!?]) +', note)
    return ' '.join(sentences[:2]) if len(sentences) > 2 else note

def generate_faq_from_body(body_text: str) -> str:
    """
    Generate 3 compact Q&As from the article body as a last-resort fallback.
    Returns markdown like:
    Q1: ...
    A:  ...
    (blank line between items)
    """
    try:
        prompt = (
            "From the article below, write exactly 3 short FAQ Q&As for beginners. "
            "Use this exact format without headings:\n"
            "Q1: <question>\nA: <answer>\n\nQ2: ...\nA: ...\n\nQ3: ...\nA: ...\n\n"
            "Keep each answer under 2 sentences.\n\nArticle:\n" + body_text[:4000]
        )
        resp = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=260, temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # ultra-safe static fallback
        return (
            "Q1: What do I need to get started?\n"
            "A: Pick one tool, try a small test task, and follow the built-in tutorial.\n\n"
            "Q2: How long before I see results?\n"
            "A: Usually within a week if you practice daily and take 1–2 paid micro-tasks.\n\n"
            "Q3: Any risks to watch for?\n"
            "A: Double-check facts, respect copyrights, and be transparent with clients."
        )

# ------------------------------
# Main daily generator
# ------------------------------
def generate_article_for_today():
    # 1) Gather & rank daily cybersecurity items
    items = fetch_cyber_feed_entries(max_per_feed=50)
    if not items:
        raise Exception("No cybersecurity feed items available today.")

    # Rank a larger pool, then select items not used yet today 
    ranked = pick_top_cyber_items(items, n=200)         # was 30
    top_items = select_unique_for_today(ranked, n=9)

    # 2) Title + brief (simple, human) — add clean edition suffix when >1 per day
    base_title = rewrite_title_for_cyber_briefing(top_items)
    existing_today_count = (CommunityArticle.query
                            .filter(CommunityArticle.date == date.today(),
                                    CommunityArticle.title.like("Daily Cybersecurity Briefing%"))
                            .count())

    headline = base_title if existing_today_count == 0 else f"{base_title} — Edition {existing_today_count + 1}"

    article_md = generate_accessible_brief(top_items)

    # 3) Auto-insert two tasteful images (top + near end)
    def _image_block(query: str, fallback_terms: str):
        url, photographer, image_page, alt = get_unsplash_image(query)
        if url:
            caption = alt or "Image via Unsplash"
            credit = f"*Photo by {photographer} on Unsplash*"
            return f"![{caption}]({url})\n{credit}\n\n"
        # Fallback if no API key
        fallback = f"https://source.unsplash.com/1600x900/?{fallback_terms}"
        return f"![Cyber image]({fallback})\n\n"

    top_img = _image_block("cybersecurity lock blue", "cybersecurity,lock,blue")
    mid_img = _image_block("laptop update security", "update,laptop,security")

    # Place one at top, one after the first major section
    article_md = f"{top_img}{article_md}\n{mid_img}"

    # 4) Tidy (keep your helpers)
    article_md = re.sub(r'(?im)^\s*faq\s*$', '', article_md).strip()
    article_md = re.sub(r'(?m)([^\n])\n##', r'\1\n\n##', article_md)
    article_md = strip_unwanted_bold(remove_gpt_dashes(article_md))

    # 5) Optional: reel script
    try:
        reel_script = generate_reel_script_veed(article_md, headline)
    except NameError:
        reel_script = ""

    # 6) Save
    html_content = markdown(article_md, extras=["tables"])
    filename = f"{datetime.now().strftime('%Y%m%d')}_{re.sub('[^a-zA-Z0-9]+', '-', headline)[:50]}"
    if CommunityArticle.query.filter_by(filename=filename).first():
        filename += f"_{random.randint(100,999)}"

    meta_title = headline
    meta_description = (
        re.sub(r'<.*?>', '', html_content)[:155] or
        "Quick daily cyber brief: 7–10 items with practical actions."
    )

    save_article_db(
        headline, article_md, filename,
        html_content=html_content,
        meta_title=meta_title,
        meta_description=meta_description,
        reel_script=reel_script
    )
    print(f"✅ Saved Cyber Briefing: {filename}")
    return filename

# ------------------------------
# DB save & routes (kept compatible)
# ------------------------------
def save_article_db(title, content_md, filename, html_content=None, meta_title=None, meta_description=None, reel_script=None):
    article = CommunityArticle(
        date=date.today(),
        filename=filename,
        title=title,
        content=content_md,
        html_content=html_content or markdown(content_md),
        meta_title=meta_title,
        meta_description=meta_description,
        reel_script=reel_script
    )
    db.session.add(article)
    db.session.commit()
    return article.id

@bp.route("/")
def show_articles():
    articles = (CommunityArticle.query
            .order_by(CommunityArticle.date.desc(), CommunityArticle.id.desc())
            .all())
    return render_template("reddit_articles.html", articles=articles)

@bp.route("/generate", methods=["POST"])
def generate():
    fname = generate_article_for_today()
    return jsonify({"filename": fname, "success": True})

@bp.route("/articles")
def published_articles():
    import re as _re
    articles = (CommunityArticle.query
            .order_by(CommunityArticle.date.desc(), CommunityArticle.id.desc())
            .all())
    for a in articles:
        source = (a.content or a.html_content or "")
        # Remove images
        plain = _re.sub(r'\!\[.*?\]\(.*?\)', '', source)
        # Remove links but keep text
        plain = _re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plain)
        # Remove bold/italic markers
        plain = _re.sub(r'\*\*|\*|__|_', '', plain)
        # Remove Markdown headings (##, ###, etc.)
        plain = _re.sub(r'^\s*#{1,6}\s*', '', plain, flags=_re.MULTILINE)
        # Remove "Photo by ..." credits
        plain = _re.sub(r'Photo by .*? on Unsplash', '', plain, flags=_re.IGNORECASE)

        words = plain.split()
        excerpt = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
        a.excerpt = excerpt   # template uses `article.excerpt`
        a.snippet = excerpt   # keep for backward-compat

    return render_template(
        "published_articles.html",
        articles=articles,
        is_plus=current_user_is_plus(),
    )

@bp.route("/articles/<filename>")
def read_article(filename):
    article = CommunityArticle.query.filter_by(filename=filename).first_or_404()
    is_plus = current_user_is_plus()

    def remove_first_heading(md_text):
        lines = (md_text or "").splitlines()
        found = False
        output = []
        for line in lines:
            if not found and line.strip().startswith("#"):
                found = True
                continue
            output.append(line)
        return "\n".join(output)

    cleaned_md = remove_first_heading(article.content or "")
    hero_url, hero_credit = _extract_hero_info(article.content or "")
    preview_summary, preview_bullets = _build_preview(article, cleaned_md)
    html_content = markdown(cleaned_md, extras=["tables"]) if is_plus else None

    meta_title = article.meta_title or article.title
    if article.meta_description:
        meta_description = article.meta_description
    else:
        meta_description = preview_summary[:160]

    # NEW: fetch newest article to power the CTA
    latest_article = CommunityArticle.query.order_by(CommunityArticle.date.desc()).first()

    return render_template(
        "single_article.html",
        title=article.title,
        date=article.date,
        content_full=html_content if is_plus else None,
        preview_summary=preview_summary,
        preview_bullets=preview_bullets,
        hero_url=hero_url,
        hero_credit=hero_credit,
        meta_title=meta_title,
        meta_description=meta_description,
        latest_article=latest_article,          # <-- pass it
        current_filename=article.filename,      # (optional) for self-check in template
        is_plus=is_plus,
    )

# Manual verification checklist:
# - Visit /all-articles/articles while logged out: list renders excerpts with “Preview briefing” buttons and Plus badge.
# - Open any article while logged out: preview renders (title/date/hero + overview summary/bullets) and locked card shows Plus CTA/back link; full text not present in page source.
# - Open an article while Plus (or after toggling is_plus flag): full content renders with no lock card.
# - Ensure latest briefing CTA at page bottom still works and list remains accessible to all users.

# CLI run
if __name__ == "__main__":
    from app import create_app
    app = create_app()
    with app.app_context():
        try:
            print("Generating today's Daily Cybersecurity Briefing...")
            fname = generate_article_for_today()
            print(f"✅ Generated and saved: {fname}")
        except Exception as e:
            import traceback
            print("❌ Generation failed:", e)
            traceback.print_exc()
            raise
