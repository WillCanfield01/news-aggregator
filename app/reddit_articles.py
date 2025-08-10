import os
import re
import json
import random
import difflib
import requests
import praw
from datetime import datetime, date, timedelta
import openai
from flask import Blueprint, render_template, jsonify
from markdown2 import markdown
from unidecode import unidecode
from app.models import CommunityArticle
from app import db
from rapidfuzz import fuzz

# ------------------------------
# Config & constants
# ------------------------------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
ADD_PERSONAL_NOTES = False  # set True if you want exactly one clean note
RECENT_WINDOW = 10  # last 10 days
VARIETY_PENALTY = 40

openai.api_key = OPENAI_API_KEY

bp = Blueprint("all_articles", __name__, url_prefix="/all-articles")

ARTICLES_DIR = os.path.join(os.path.dirname(__file__), "generated_articles")
os.makedirs(ARTICLES_DIR, exist_ok=True)

TARGET_SUBS = ["sidehustle", "Entrepreneur", "AItools"]  # <- NEW
TIME_FILTER = "day"
POSTS_PER_SUB = 15  # scan enough to find a solid AI-money angle

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
    has_it_banned = any(bad in t for bad in IT_BANNED_TOPICS)
    return has_tool and has_action and not has_it_banned

def is_monetizable_side_gig(title):
    t = title.lower()
    # must contain a tool + action
    if not contains_tool_and_action(t):
        return False
    # must NOT be enterprise IT
    if any(bad in t for bad in IT_BANNED_TOPICS):
        return False
    return True

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
# Reddit → pick best AI money idea
# ------------------------------
def fetch_candidates_from_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="RealRoundup/AI-Money-Daily 1.0"
    )
    results = []
    for sub in TARGET_SUBS:
        try:
            sr = reddit.subreddit(sub)
            for post in sr.top(time_filter=TIME_FILTER, limit=POSTS_PER_SUB):
                if getattr(post, "over_18", False):
                    continue
                title = post.title or ""
                selftext = post.selftext or ""
                if not (is_safe(title) and is_safe(selftext)):
                    continue
                results.append({
                    "sub": sub,
                    "id": post.id,
                    "title": title,
                    "selftext": selftext,
                    "score": getattr(post, "score", 0),
                    "url": f"https://reddit.com{getattr(post, 'permalink', '')}"
                })
        except Exception:
            continue
    return results

def monetization_variety_penalty(candidate_method):
    recent_methods = [
        detect_monetization_method(a.title)
        for a in CommunityArticle.query.order_by(CommunityArticle.date.desc()).limit(RECENT_WINDOW)
    ]
    recent_methods = [m for m in recent_methods if m]
    return VARIETY_PENALTY if candidate_method in recent_methods else 0

def get_reddit_client():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="RealRoundup/AI-Money-Daily 1.0"
    )

def score_candidate(c):
    t = (c["title"] + " " + c.get("selftext", "")).lower()
    ai_hits = sum(1 for k in AI_KEYWORDS if k in t)
    base = c["score"] + ai_hits * 50
    length_penalty = max(0, len(c["title"]) - 120) // 10
    method = detect_monetization_method(t)
    score = base - length_penalty - monetization_variety_penalty(method)
    return score

def get_best_ai_money_post():
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="daily_ai_money"
    )

    # fixed spelling + combined subs
    subreddit = reddit.subreddit("sidehustle+Entrepreneur+ArtificialIntelligence+ChatGPT")
    posts = subreddit.top(time_filter="day", limit=50)

    candidates = []
    for post in posts:
        title = (post.title or "").strip()
        body = (post.selftext or "").strip()

        if is_generic(title) or is_generic(body):
            continue
        if not contains_tool_and_action(f"{title} {body}"):
            continue

        candidates.append({
            "title": title,
            "selftext": body,        # ← include for downstream use
            "score": getattr(post, "score", 0),
            "url": f"https://reddit.com{getattr(post, 'permalink', '')}",
            "id": post.id
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0] if candidates else None

# ------------------------------
# Generation prompts (new format)
# ------------------------------
def rewrite_title_for_ai_money(original_title: str, seed_text: str = "") -> str:
    """
    Generate specific, helpful, non-generic titles and pick one that:
    - names an exact tool/tech or niche
    - states the outcome/benefit
    - calls out audience/use case
    - avoids generic phrases and past-title near-duplicates
    - stays under 80 chars (soft 72–78 sweet spot)
    """
    banned_phrases = [
        "beginner’s guide", "beginners guide", "how to use ai", "start earning",
        "from scratch", "ways to make money", "how to make money", "side hustle",
        "friendly guide", "build confidence"
    ]

    # Ask the model for multiple concrete options so we can pick the best
    prompt = f"""
You are creating article titles for a daily blog helping everyday people find
new AI-powered side gigs they can start this week.

Rules:
- Each title must clearly describe a monetizable AI use case for beginners.
- Must include BOTH:
    1. A specific AI tool, model, or platform by name
       (e.g., ChatGPT, Midjourney, DALL·E, ElevenLabs, Perplexity, Stable Diffusion)
    2. An income-related activity or deliverable
       (e.g., selling Etsy printables, writing product descriptions, creating YouTube shorts)
- Keep the focus on small-scale, realistic side hustles — NOT enterprise IT, cybersecurity, workflows, or admin tasks.
- Avoid jargon, corporate language, or references to IT departments.
- Avoid generic phrases like: {", ".join(banned_phrases)}.
- Keep ≤ 80 characters; prefer 48–78.
- Do NOT mention Reddit, enterprise identity, IT workflow, or any topic unrelated to earning money.

Input idea: {original_title}
Seed/context (optional): {seed_text[:500]}

Return JSON: {{"titles": ["t1","t2","t3","t4","t5","t6","t7","t8"]}}
"""

    try:
        resp = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=220,
            temperature=0.7,
            response_format={"type":"json_object"}
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        candidates = [t.strip() for t in data.get("titles", []) if t.strip()]
    except Exception:
        candidates = []

    if not candidates:
        # very safe fallback
        candidates = [original_title[:78]]

    # Pull previous titles to avoid near-dupes
    past = [a.title for a in CommunityArticle.query.all()]

    def looks_generic(t: str) -> bool:
        low = t.lower()
        return any(p in low for p in banned_phrases)

    def has_specific_signal(t: str) -> int:
        low = t.lower()
        signals = 0
        signals += any(k in low for k in ["chatgpt","gpt","midjourney","stable diffusion","dall-e","elevenlabs","zapier","make.com","notion ai","perplexity","elicit","research rabbit"])
        signals += any(k in low for k in ["etsy","airbnb","realtor","local business","podcast","youtube","tiktok","shopify","webflow","wordpress"])
        signals += any(k in low for k in ["workflow","pipeline","automation","prompt","template","mvp","landing page","agent"])
        return int(signals)

    def score(t: str) -> float:
        # higher is better
        length = len(t)
        length_pen = 0 if 48 <= length <= 78 else abs(63 - min(length, 80)) * 0.6
        spec = has_specific_signal(t)
        # similarity penalty vs past titles
        dup_pen = max((fuzz.ratio(t.lower(), p.lower()) for p in past), default=0)
        return spec*10 - length_pen - (dup_pen > 85)*20 - (looks_generic(t))*15

    # Filter & pick best
    filtered = [t for t in candidates if not looks_generic(t)]
    if not filtered:
        filtered = candidates

    best = max(filtered, key=score)
    # Final trim/sanitize
    best = best.strip().strip('"').strip("'")
    return best[:80]

def generate_outline_for_ai_money(topic: str, seed_text: str):
    """
    Force a consistent, action-first outline. No Reddit references.
    """
    prompt = (
    f"You are writing a daily guide called 'One New Way to Make Money With AI: {topic}'. "
    "Rules:\n"
    "- Focus on ONE specific AI use case, not general freelancing.\n"
    "- Include exact tools (by name), where to get them (with placeholder URLs), and free/paid options.\n"
    "- Give numbered setup steps that a total beginner could follow today, including menu clicks, settings, and file formats.\n"
    "- Include at least ONE real-world example, case study, or mini scenario showing how someone actually earned money this way.\n"
    "- Include realistic earnings figures from known marketplaces.\n"
    "- Include at least ONE unique tip or trick not found in generic blog posts.\n"
    "- Do not use phrases like 'build confidence', 'start small', or 'be patient' unless directly tied to the method.\n"
    "- Avoid generic side hustle language. Make it actionable and specific.\n"
    "- No social media sourcing or Reddit references.\n"
    "- Sections:\n"
    "  1) Hook: Why this specific AI use case works now\n"
    "  2) Tools you need (name + placeholder link + cost)\n"
    "  3) Exact setup steps (numbered, 7–10 steps)\n"
    "  4) Example earning scenario (numbers)\n"
    "  5) Pitfalls & solutions\n"
    "  6) Scaling & automation tips\n"
    "  7) Quick checklist\n"
    "  8) FAQ (3 items)\n"
    "Return in Markdown headings and bullet points."
)
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=420, temperature=0.4
    )
    return resp.choices[0].message.content

def generate_article_body(topic: str, outline_md: str, seed_text: str):
    prompt = (
        "Write a 700–900 word, beginner-friendly guide based on the outline below.\n"
        "Tone: clear, calm, encouraging. No fluff, no corporate-speak. No mention of Reddit.\n"
        "Include: Hook, Tools, Steps, Pricing/Earnings, Pitfalls, Scale, Checklist, FAQ (3 Q&As).\n"
        f"Topic: {topic}\n\n"
        f"Seed notes (optional context from source):\n{seed_text[:800]}\n\n"
        f"Outline:\n{outline_md}\n\n"
        "Article:"
    )
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500, temperature=0.45,
    )
    return resp.choices[0].message.content

def generate_reel_script(article_text: str, topic: str):
    prompt = (
        "Write a 30–40s vertical video script with:\n"
        "- Cold hook in first 2s.\n"
        "- 3 numbered beats showing how to make money with this AI idea.\n"
        "- Simple words; no jargon.\n"
        "- Strong CTA: 'Read the full guide at TheRealRoundup.com'.\n"
        "- On-screen text cues in [BRACKETS].\n"
        f"Topic: {topic}\n\nSource:\n{article_text[:1200]}\n\nScript:"
    )
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=260, temperature=0.7
    )
    return resp.choices[0].message.content.strip()

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
    post = get_best_ai_money_post()
    if not post:
        # fallback to your older ranking method or bail gracefully
        candidates = fetch_candidates_from_reddit()
        if not candidates:
            raise Exception("No suitable Reddit posts found this day.")
        post = sorted(candidates, key=score_candidate, reverse=True)[0]

    cleaned_topic = clean_title(post["title"])
    headline = rewrite_title_for_ai_money(cleaned_topic, post.get("selftext",""))

    outline = generate_outline_for_ai_money(headline, post.get("selftext", ""))
    article_md = generate_article_body(headline, outline, post.get("selftext", ""))
    article_md = re.sub(r'(?i)\*\*?\s*personal\s*note\s*:\s*.*?(?:\n|$)', '', article_md)

    # 3) Light post-processing
    #    (Keep your personal-notes quirks and cleanup)
    sections = split_markdown_sections(article_md)
    article_with_human = ""
    img_count = 0
    faq_section = None

    for i, (heading, content) in enumerate(sections):
        lower = heading.lower()
        is_faq = lower.startswith("faq")
        is_conclusion = lower.startswith("conclusion")  # may exist depending on model

        if not (is_faq or is_conclusion):
            article_with_human += f"## {heading}\n\n"

        if (
            img_count < 4 and content and len(content.strip()) > 60
            and not is_faq and not is_conclusion
        ):
            suggestion = generate_section_image_suggestion(
                headline, [ "make money with ai", "ai side hustle", "automation" ],
                outline, heading, content
            )
            if suggestion:
                image_url, photographer, image_page, unsplash_alt = get_unsplash_image(suggestion.get("query", ""))
                caption = unsplash_alt or suggestion.get("caption", "Illustrative image")
                if image_url:
                    article_with_human += f"![{caption}]({image_url})\n*Photo by {photographer} on Unsplash*\n\n"
                    img_count += 1

        # Trim duplicated heading echoes
        lines = content.splitlines()
        if lines:
            first = re.sub(r'\W+', '', (lines[0] or "").lower())
            hcmp = re.sub(r'\W+', '', heading.lower())
            if hcmp and hcmp in first:
                lines = lines[1:]
        body = "\n".join(lines).strip()
        article_with_human += body + "\n\n"

        if is_faq:
            faq_section = (heading, content)
            continue

    # --- Normalize FAQ: remove stray inline Q/A, then add exactly one FAQ ---
    inline_qa_pat = r'(?mis)(^|\n)\s*(\**Q\d+:\s.*?)(?=(?:\n\s*\**Q\d+:|\n##\s|\Z))'
    inline_qas = [m.group(2).lstrip('*').strip() for m in re.finditer(inline_qa_pat, article_with_human)]
    article_with_human = re.sub(inline_qa_pat, r'\1', article_with_human).strip()

    synth_faq = ""
    if not faq_section and inline_qas:
        synth_faq = "\n\n".join(inline_qas)

    already_has_faq = re.search(r'(?mi)^\s*##\s*FAQ\b', article_with_human)

    if not already_has_faq:
        if faq_section:
            article_with_human += "\n## FAQ\n\n" + faq_section[1].strip() + "\n\n"
        elif synth_faq:
            article_with_human += "\n## FAQ\n\n" + synth_faq.strip() + "\n\n"
        else:
            gen_faq = generate_faq_from_body(article_with_human)
            if gen_faq:
                article_with_human += "\n## FAQ\n\n" + gen_faq.strip() + "\n\n"

    if not is_monetizable_side_gig(headline):
        print(f"⚠️ Rejected headline '{headline}' — not a monetizable everyday side gig.")
        return None

    # Layout normalization (spacing + stray 'FAQ' lines)
    article_with_human = re.sub(r'(?im)^\s*faq\s*$', '', article_with_human).strip()
    article_with_human = re.sub(r'(?m)([^\n])\n##', r'\1\n\n##', article_with_human)
    article_with_human = remove_gpt_dashes(article_with_human)
    article_with_human = strip_unwanted_bold(article_with_human)

    # 4) Reel script
    reel_script = generate_reel_script(article_with_human, headline)

    # 5) Save
    html_content = markdown(article_with_human)
    filename = f"{datetime.now().strftime('%Y%m%d')}_{re.sub('[^a-zA-Z0-9]+', '-', headline)[:50]}"
    if CommunityArticle.query.filter_by(filename=filename).first():
        print(f"⚠️ Article for filename {filename} already exists. Skipping save.")
        return filename

    # Basic meta
    meta_title = headline
    meta_description = (re.sub(r'<.*?>', '', html_content)[:155] or
                        "Daily, beginner-friendly ways to make money with AI.")

    if is_duplicate(headline):
        print("⚠️ Headline is a duplicate after generation. Aborting today’s article.")
        return None

    save_article_db(
        headline, article_with_human, filename,
        html_content=html_content,
        meta_title=meta_title,
        meta_description=meta_description,
        reel_script=reel_script
    )
    print(f"✅ Saved to DB: {filename}")
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
    articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
    return render_template("reddit_articles.html", articles=articles)

@bp.route("/generate", methods=["POST"])
def generate():
    fname = generate_article_for_today()
    return jsonify({"filename": fname, "success": True})

@bp.route("/articles")
def published_articles():
    import re as _re
    articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
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
    html_content = markdown(cleaned_md)

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
            print("Generating today's AI money article...")
            fname = generate_article_for_today()
            print(f"✅ Generated and saved: {fname}")
        except Exception as e:
            import traceback
            print("❌ Generation failed:", e)
            traceback.print_exc()
            raise