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
from app import db
from rapidfuzz import fuzz
import feedparser
from urllib.parse import urlparse

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
    """Create a simple, dated briefing title."""
    today = datetime.utcnow().strftime("%B %d, %Y")
    # pull 1–2 strongest keywords for flavor if present
    top = items[:3]
    hints = []
    for it in top:
        t = (it["title"] or "").lower()
        if "cve-" in t and "CVE" not in hints:
            hints.append("CVE alerts")
        if "ransomware" in t and "Ransomware" not in hints:
            hints.append("Ransomware")
        if "zero-day" in t or "zero day" in t:
            if "Zero-day" not in hints:
                hints.append("Zero-day")
    suffix = f" — {', '.join(hints)}" if hints else ""
    return f"Daily Cybersecurity Briefing ({today}){suffix}"

def generate_outline_for_cyber(items: list) -> str:
    """Ask the model for a structured outline based on the top items."""
    # compact context to keep tokens modest
    bullet_ctx = "\n".join([f"- {it['title']} ({it['source']}) — {it['link']}" for it in items])
    prompt = f"""
You're an editor writing a concise daily cybersecurity briefing for busy professionals.

Use ONLY the context links provided below. You may paraphrase, synthesize, and extract key facts,
but do NOT invent facts that aren't supported by the items. Cite sources inline like [source: domain].

Context items (titles + links):
{bullet_ctx}

Return a tight Markdown outline with these sections and nothing else:
1) Top Stories (3–5 bullets; each bullet ends with [source: domain])
2) Critical CVEs & Patches (table: CVE | Severity | Affected | Fix/Workaround | Source)
3) Ransomware & Threat Activity (2–4 bullets)
4) What This Means (why it matters in plain language)
5) Quick Actions for Security Teams (5–7 checklist items)
6) Sources (bulleted list of the links with domains)
"""
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=700,
        temperature=0.2
    )
    return resp.choices[0].message.content

def generate_cyber_article_body(outline_md: str) -> str:
    """Expand the outline into a 700–900 word brief with tables kept valid."""
    prompt = f"""
Expand the following outline into a 700–900 word daily cybersecurity brief.
Rules:
- Keep the exact section order from the outline.
- Keep the CVE table exactly 5 columns: | CVE | Severity | Affected | Fix/Workaround | Source |
- Use plain, professional tone. No hype. No filler.
- Attribute facts to sources inline like [source: domain], and list all links in a final "Sources" section.
- Do not copy long passages verbatim; paraphrase instead.

Outline:
{outline_md}

Article:
"""
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=1500,
        temperature=0.35
    )
    return resp.choices[0].message.content

def generate_reel_script_veed(article_text: str, topic: str):
    """
    30–40s vertical script variants for a daily cyber brief.
    """
    prompt = f"""
Create 5 short vertical video script VARIANTS for a Daily Cybersecurity Brief.
Each variant:
- 30–40 seconds (~90–110 words)
- First line [HOOK] ≤ 8 words, curiosity-based, no hype
- Short lines (≤ 8 words) for captions
- Name 1–2 concrete items (e.g., CVE id, vendor patch) if present
- Honest tone: “not instant, but here’s the takeaway”
- End with [CTA] “Full brief at TheRealRoundup.com”

Topic: {topic}

Reference (use only as context; don't quote verbatim):
{article_text[:1200]}

Return JSON: {{"variants": ["...","...","...","...","..."]}}
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.5,
            max_tokens=900,
            response_format={"type":"json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        variants = [v.strip() for v in data.get("variants", []) if v.strip()]
    except Exception:
        variants = []

    if not variants:
        return (
            "Today’s biggest cyber risks, fast\n"
            "[B-ROLL: scrolling security dashboard]\n"
            "Zero-day patched in a major browser\n"
            "Two ransomware groups hit healthcare\n"
            "[B-ROLL: news headlines montage]\n"
            "If you manage endpoints, patch today\n"
            "Block known C2 IPs and rotate creds\n"
            "[B-ROLL: terminal / firewall rules]\n"
            "[CTA] Full brief at TheRealRoundup.com"
        )

    best = max(variants, key=_score_reel_variant)
    return best

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
    items = fetch_cyber_feed_entries(max_per_feed=25)
    if not items:
        raise Exception("No cybersecurity feed items available today.")
    top_items = pick_top_cyber_items(items, n=8)

    # 2) Title + outline + article
    headline = rewrite_title_for_cyber_briefing(top_items)
    if is_duplicate(headline):
        # small suffix to avoid filename collision if multiple runs same day
        headline += " (Update)"

    outline = generate_outline_for_cyber(top_items)
    article_md = generate_cyber_article_body(outline)

    # 3) Normalize / cleanup (keep your existing FAQ helper)
    sections = split_markdown_sections(article_md)
    article_norm = ""
    faq_section = None

    for heading, content in sections:
        lower = (heading or "").lower()
        is_faq = lower.startswith("faq")
        if not is_faq:
            article_norm += f"## {heading}\n\n"
        article_norm += (content or "") + "\n\n"
        if is_faq:
            faq_section = (heading, content)

    # Ensure we have exactly one FAQ (reuse your helper)
    inline_qa_pat = r'(?mis)(^|\n)\s*(\**Q\d+:\s.*?)(?=(?:\n\s*\**Q\d+:|\n##\s|\Z))'
    inline_qas = [m.group(2).lstrip('*').strip() for m in re.finditer(inline_qa_pat, article_norm)]
    article_norm = re.sub(inline_qa_pat, r'\1', article_norm).strip()

    already_has_faq = re.search(r'(?mi)^\s*##\s*FAQ\b', article_norm)
    if not already_has_faq:
        if faq_section:
            article_norm += "\n## FAQ\n\n" + faq_section[1].strip() + "\n\n"
        elif inline_qas:
            article_norm += "\n## FAQ\n\n" + "\n\n".join(inline_qas).strip() + "\n\n"
        else:
            gen_faq = generate_faq_from_body(article_norm)
            if gen_faq:
                article_norm += "\n## FAQ\n\n" + gen_faq.strip() + "\n\n"

    # Final tidy
    article_norm = re.sub(r'(?im)^\s*faq\s*$', '', article_norm).strip()
    article_norm = re.sub(r'(?m)([^\n])\n##', r'\1\n\n##', article_norm)
    article_norm = strip_unwanted_bold(remove_gpt_dashes(article_norm))

    # 4) Optional: reel script
    reel_script = generate_reel_script_veed(article_norm, headline)

    # 5) Save
    html_content = markdown(article_norm, extras=["tables"])
    filename = f"{datetime.now().strftime('%Y%m%d')}_{re.sub('[^a-zA-Z0-9]+', '-', headline)[:50]}"
    if CommunityArticle.query.filter_by(filename=filename).first():
        # ensure unique filename if identical title/date
        filename += f"_{random.randint(100,999)}"

    meta_title = headline
    meta_description = (
        re.sub(r'<.*?>', '', html_content)[:155] or
        "Daily cybersecurity briefing: top stories, critical CVEs, ransomware activity, and quick actions."
    )

    save_article_db(
        headline, article_norm, filename,
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
        a.snippet = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
    return render_template("published_articles.html", articles=articles)

@bp.route("/articles/<filename>")
def read_article(filename):
    article = CommunityArticle.query.filter_by(filename=filename).first_or_404()

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
    html_content = markdown(cleaned_md, extras=["tables"])

    meta_title = article.meta_title or article.title
    if article.meta_description:
        meta_description = article.meta_description
    else:
        plain = re.sub(r'<.*?>', '', html_content)
        meta_description = plain[:160]

    # NEW: fetch newest article to power the CTA
    latest_article = CommunityArticle.query.order_by(CommunityArticle.date.desc()).first()

    return render_template(
        "single_article.html",
        title=article.title,
        date=article.date,
        content=html_content,
        meta_title=meta_title,
        meta_description=meta_description,
        latest_article=latest_article,          # <-- pass it
        current_filename=article.filename       # (optional) for self-check in template
    )

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