# app/scripts/generate_timeline_round.py
from __future__ import annotations

import os
import re
import random
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from flask import current_app
from app.roulette.models import TimelineRound, TimelineGuess
import pytz
import requests
from sqlalchemy import select

from app.extensions import db
from app.roulette.models import TimelineRound
try:
    from app.roulette.icon_ai import pick_icon_for_text  # optional
except Exception:  # pragma: no cover
    def pick_icon_for_text(_: str) -> str:
        # simplest safe fallback icon name that exists in /static/roulette/icons/
        return "star.svg"
    
TZ = pytz.timezone(os.getenv("TIME_ZONE", "America/Denver"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

OAI_URL = "https://api.openai.com/v1/chat/completions"  # works with o3-mini/gpt-4o-mini, etc.
OAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

WIKI_URL = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{m}/{d}"

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

def _words(s: str) -> list[str]:
    return [w for w in WORD_RE.findall(s or "")]

def _wlen(s: str) -> int:
    return len(_words(s))

# Domains and nostalgia signals for younger / TikTok crowd
POP_KEYWORDS = {
    "social_media": ["tiktok","instagram","facebook","twitter","x.com","snapchat","youtube","reddit","twitch","vine","myspace"],
    "gaming": ["nintendo","playstation","xbox","pokemon","minecraft","fortnite","roblox","steam","esports","blizzard","sony"],
    "music": ["spotify","itunes","mtv","grammy","billboard","taylor swift","drake","bts","k-pop","eminem","nirvana"],
    "film_tv": ["netflix","disney","marvel","star wars","hbo","pixar","oscars","hulu","prime video","anime"],
    "tech": ["iphone","android","apple","google","microsoft","ai","openai","tesla","spacex","nvidia","samsung","internet","www"],
    "sports": ["nba","nfl","mlb","nhl","fifa","world cup","olympics","super bowl","lakers","yankees","patriots"],
    "internet_culture": ["meme","viral","hashtag","emoji","stream","podcast","blog","wiki","open source","linux","browser"],
    "travel": ["ship","liner","cruise","harbor","harbour","port","airport","airline","flight","railway","train","metro","tram","bus","ferry","voyage"],
}

NEGATIVE_TERMS = [
    "killed", "killing", "dead", "dies", "death", "fatal",
    "massacre", "genocide", "famine",
    "bomb", "bombing", "bombings", "explosion", "blast",
    "attack", "attacks", "assault", "raid", "airstrike", "suicide bomber", "suicide attack", "suicide bombing",
    "war", "battle", "invasion", "occupation", "uprising", "coup",
    "shooting", "shootings", "mass shooting", "gunman", "terror", "terrorism", "terrorist", "hostage",
    "crash", "collides", "collision", "derailment",
    "earthquake", "hurricane", "typhoon", "tsunami", "flood", "wildfire",
    "eruption", "eruptions", "erupts", "volcano", "volcanic",
]

BANNED_JARGON = {
    "prototype", "initiative", "specialized", "project", "institute",
    "laboratory", "pioneers", "researchers", "data processing",
}


# --------------------------- utilities ---------------------------

def _today_local_date():
    return datetime.now(TZ).date()


def _http_get_json(url: str, headers: dict | None = None, params: dict | None = None) -> dict:
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()

def _soften_real_title(title: str) -> str:
    """
    Light normalization so the real headline remains human and specific.
    - Trim whitespace / stray code fences
    - Avoid hyper-precise numbers only if they look like big counts
    - Do NOT replace proper nouns (keeps it believable)
    """
    if not title:
        return ""

    t = str(title).strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t, flags=re.MULTILINE)  # strip code fences

    # collapse very large counts like 12,345 or 2000000 → “thousands”
    t = re.sub(r"\b(\d{1,3}(?:,\d{3}){1,}|\d{5,})\b", "thousands", t)

    # normalize doubled determiners that sometimes sneak in
    t = re.sub(r"\b(?:the\s+the|a\s+a|an\s+an|the\s+a|a\s+the)\b", lambda m: m.group(0).split()[0], t, flags=re.I)

    # cleanup spaces
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _sentence_case(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]

def _sanitize_sentence(s: str) -> str:
    """
    Remove immediate duplicates, squeeze spaces, and ensure trailing period.
    """
    tokens = _words(s)
    cleaned: list[str] = []
    for w in tokens:
        if cleaned and cleaned[-1].lower() == w.lower():
            continue
        cleaned.append(w)
    if not cleaned:
        return s
    text = " ".join(cleaned)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text.endswith("."):
        text += "."
    return text

def _is_tragedy(text: str) -> bool:
    lc = (text or "").lower()
    return any(w in lc for w in NEGATIVE_TERMS)

def _infer_domain(text: str) -> str:
    if not text:
        return "general"
    lc = text.lower()
    for domain, words in POP_KEYWORDS.items():
        if any(w in lc for w in words):
            return domain
    return "general"

def _score_for_youth(event_text: str, year: Optional[int]) -> float:
    """
    Higher score for modern/pop-culture domains and youth nostalgia years.
    """
    if not event_text:
        return -1e9
    if _is_tragedy(event_text):
        return -1e9

    score = 0.0
    domain = _infer_domain(event_text)
    pop_domains = {"tech","social_media","gaming","music","film_tv","internet_culture","sports"}
    if domain in pop_domains:
        score += 4.0

    lc = event_text.lower()
    for words in POP_KEYWORDS.values():
        if any(w in lc for w in words):
            score += 1.5
            break

    try:
        if year is not None:
            if 1995 <= year <= 2012:
                score += 4.0
            elif 1985 <= year <= 1994:
                score += 3.0
            elif 2013 <= year <= 2018:
                score += 2.0
            elif year < 1975:
                score -= 2.5
    except Exception:
        pass
    # brevity
    L = len(event_text)
    if 40 <= L <= 180:
        score += 1.0
    elif L > 220:
        score -= 0.5
    return score

def _fit_length(text: str, min_words: int, max_words: int) -> str:
    """
    Clamp to a word window, padding lightly if short.
    """
    words = _words(text)
    if not words:
        return text or ""
    # Do not pad with stock fillers; rely on generator to meet the range
    if len(words) > max_words:
        words = words[:max_words]
        awkward = {"for","of","and","to","in","as","was","were","by","with","at"}
        while len(words) > min_words and words[-1].lower() in awkward:
            words.pop()
    s = " ".join(words).rstrip(" .,;")
    if not s.endswith("."):
        s += "."
    return s

def _strip_jargon(s: str) -> str:
    text = s
    for bad in BANNED_JARGON:
        text = re.sub(rf"\b{re.escape(bad)}\b", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()

def _normalize_choice(text: str, min_words: int, max_words: int) -> str:
    def _one_sentence(s: str) -> str:
        parts = [p.strip() for p in re.split(r"[.!?]+", s) if p.strip()]
        return parts[0] if parts else s

    cleaned = _strip_jargon(text)
    cleaned = _one_sentence(cleaned)
    cleaned = _sanitize_sentence(cleaned)
    cleaned = _fit_length(cleaned, min_words, max_words)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return _sentence_case(cleaned)

def _strip_years(text: str) -> str:
    return re.sub(r"\b(19|20)\d{2}\b", "", text or "")

def _mostly_year_swap(a: str, b: str) -> bool:
    ta = [w for w in _words(_strip_years(a).lower()) if w]
    tb = [w for w in _words(_strip_years(b).lower()) if w]
    if not ta or not tb:
        return False
    sa, sb = set(ta), set(tb)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union > 0.75

def _normalize_tokens(s: str) -> list[str]:
    return [w.lower() for w in _words(s)]

def _ngram_set(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def _too_similar(a: str, b: str) -> bool:
    ta = _normalize_tokens(a)
    tb = _normalize_tokens(b)
    if not ta or not tb:
        return False

    # identical starts/ends
    if ta[:6] == tb[:6] or ta[-6:] == tb[-6:]:
        return True

    sa, sb = set(ta), set(tb)

    # one mostly contained in the other
    a_text = " ".join(ta)
    b_text = " ".join(tb)
    if a_text in b_text or b_text in a_text:
        return True

    # shared 3-grams
    tri_a = _ngram_set(ta, 3)
    tri_b = _ngram_set(tb, 3)
    if tri_a and tri_b and tri_a.intersection(tri_b):
        return True

    # shared 2-grams, allow a few, block heavy overlap
    bi_a = _ngram_set(ta, 2)
    bi_b = _ngram_set(tb, 2)
    if bi_a and bi_b:
        overlap = len(bi_a & bi_b)
        if overlap >= 3:
            return True

    # simple Jaccard on words
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    jacc = inter / union
    return jacc > 0.5

def _remix_structure(text: str, rng: random.Random) -> str:
    """
    Lightly reorders the sentence to avoid cloned openings.
    """
    s = text.strip().rstrip(".")
    core = s[0].lower() + s[1:] if s and s[0].isupper() else s
    templates = [
        "{core}",
        "Around that time, {core}",
        "Later that week, {core}",
        "{core}, drawing a lot of attention",
        "People were talking everywhere when {core}",
    ]
    pick = rng.choice(templates).format(core=core)
    pick = pick.strip()
    if not pick.endswith("."):
        pick += "."
    return _sentence_case(pick)

def _domain_ok(text: str, domain: str) -> bool:
    return True  # loosened: allow variety for decoys

def _domain_flair(domain: str, year_hint: Optional[int] = None) -> list[str]:
    classic = {
        "tech": [
            "after coverage in trade papers",
            "as radio shows explained the breakthrough",
            "with newspapers highlighting the debut",
        ],
        "film_tv": [
            "after a packed theater premiere",
            "as viewers tuned in on broadcast",
            "with critics discussing the release",
        ],
        "music": [
            "after the record climbed radio charts",
            "as fans lined up for the album",
            "with magazine write-ups",
        ],
        "sports": [
            "during a headline season",
            "as crowds filled the stands",
            "with highlight reels on nightly news",
        ],
        "general": [
            "covered by major papers",
            "picked up by radio bulletins",
            "noted in news roundups",
        ],
    }
    modern = {
        "tech": [
            "after hype on launch week",
            "as fans queued online",
            "during a big live stream",
            "with creators posting reactions",
        ],
        "social_media": [
            "as timelines flooded with posts",
            "after a hashtag trended",
            "with creators jumping in",
            "as clips spread everywhere",
        ],
        "gaming": [
            "after a record-breaking tournament",
            "as servers filled with players",
            "during a massive update",
            "while streamers highlighted it",
        ],
        "music": [
            "ahead of a sold-out tour",
            "after the single hit the charts",
            "as fans shared clips",
            "during a surprise release",
        ],
        "film_tv": [
            "after a buzzy premiere",
            "as watch parties popped up",
            "while fans quoted scenes",
            "during a viral trailer drop",
        ],
        "sports": [
            "during a heated season",
            "as crowds packed arenas",
            "after a clutch finish",
            "with highlight reels everywhere",
        ],
        "internet_culture": [
            "as memes took over feeds",
            "after forums lit up",
            "with gifs shared nonstop",
            "as blogs covered it",
        ],
        "general": [
            "covered by major outlets",
            "picked up by headlines",
            "noted in news recaps",
            "remembered in year-end lists",
        ],
    }
    is_modern = year_hint is None or year_hint >= 1995
    table = modern if is_modern else classic
    return table.get(domain, table.get("general", ["covered by major outlets"]))

def _extract_year_hint(text: str) -> Optional[int]:
    """
    Try to pull a 4-digit year from the real event to guide fake era.
    """
    if not text:
        return None
    years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
    for y in years:
        try:
            val = int(y)
            if 1500 <= val <= 2099:
                return val
        except Exception:
            continue
    return None

def _openai_fakes_from_real(real_text: str, month_name: str, domain: str, min_len: int, max_len: int, year_hint: Optional[int] = None, salt: int = 0) -> Tuple[str, str]:
    """
    Produce two plausible-but-false events similar in tone/length to the real item.
    """
    real_len = _wlen(real_text)
    era_note = ""
    if year_hint:
        decade = (year_hint // 10) * 10
        era_note = f"Match the tone and era near the {decade}s."

    def _ask_openai() -> Optional[str]:
        sys_prompt = (
            "Write one believable 'On this day' style sentence (no bullet). "
            f"Length {min_len}-{max_len} words. One sentence only. No trailing fragments. "
            "No tragedies/violence. Avoid jargon. Do not copy the real event. "
            "Keep tone like a concise Wikipedia entry. Do not prepend a year dash."
        )
        if era_note:
            sys_prompt += f" {era_note}"
        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Real event for context (do not copy): {real_text}\nMonth: {month_name}\nEra: {year_hint or 'any'}\n"},
            ],
            "temperature": 0.9,
        }
        try:
            r = requests.post(
                OAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=12,
            )
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return None

    if OPENAI_API_KEY:
        fakes: list[str] = []
        for _ in range(3):
            out = _ask_openai()
            if out and min_len <= _wlen(out) <= max_len and not _is_tragedy(out):
                if all(not _too_similar(out, f) for f in fakes) and not _too_similar(out, real_text):
                    fakes.append(out)
            if len(fakes) >= 2:
                return fakes[0], fakes[1]

    # ---------- deterministic fallback ----------
    rng = random.Random(
        int(datetime.now(TZ).strftime("%Y%m%d")) ^
        (hash(real_text) & 0xFFFFFFFF) ^
        (salt & 0xFFFF)
    )

    classic_domains = ["travel","culture","business","science","music","film_tv","sports","general"]
    modern_domains = classic_domains + ["tech","social_media","internet_culture","gaming"]
    use_modern = year_hint is None or year_hint >= 1995
    base_domains = modern_domains if use_modern else classic_domains

    classic_subjects = {
        "music": ["a chart single gains momentum", "a festival lineup draws fans", "a celebrated conductor debuts a work", "a radio hit sweeps playlists", "a celebrated tour stop draws crowds"],
        "film_tv": ["a new series premiere", "a film festival screening", "a broadcast special", "a documentary debut", "a cinema release"],
        "sports": ["a championship game", "a record-setting match", "a landmark win streak", "a headline rivalry", "a marathon milestone"],
        "travel": ["a historic ship visit", "a new ferry route", "a busy harbor festival", "a landmark train run", "a city waterfront opening"],
        "culture": ["a museum opening", "a public art show", "a city parade", "a landmark restoration", "a film festival night"],
        "business": ["a flagship store opening", "a brand collaboration", "a company milestone", "a major product unveiling", "a regional expo"],
        "science": ["a planetarium reveal", "a space exhibit", "a research announcement", "a science fair highlight", "a new laboratory wing"],
        "general": ["a national celebration", "a civic ceremony", "a community milestone", "a public event", "a major exhibit"],
    }
    modern_subjects = {
        "tech": ["a streaming feature", "a phone update", "an AI tool launch", "a cloud gaming rollout", "a wearable drop"],
        "social_media": ["a hashtag challenge", "a video trend", "a live stream format", "a creator payout push", "a viral filter"],
        "gaming": ["a multiplayer update", "an esports upset", "a crossover reveal", "a record game launch", "a fan-favorite DLC"],
        "music": ["a surprise album", "a festival headliner set", "a chart single", "a viral remix", "a tour kickoff"],
        "film_tv": ["a streaming finale", "a breakout indie film", "a viral trailer", "a hit animated episode", "a fan watch party"],
        "sports": ["a buzzer-beater playoff game", "a record marathon", "a breakout rookie season", "a championship parade", "an underdog finals win"],
        "internet_culture": ["a meme wave", "a viral gif moment", "a podcast crossover", "a fandom meetup", "a blog post that blows up"],
        "travel": classic_subjects["travel"],
        "culture": classic_subjects["culture"],
        "business": classic_subjects["business"],
        "science": classic_subjects["science"],
        "general": classic_subjects["general"],
    }

    classic_actions = {
        "music": ["tops radio charts", "draws crowds to venues", "receives strong newspaper reviews", "stays on playlists", "leads entertainment pages"],
        "film_tv": ["draws theater crowds", "earns strong newspaper reviews", "airs to solid ratings", "gets noted by critics", "runs in packed cinemas"],
        "sports": ["draws national attention", "fills sports pages", "is replayed on nightly news", "sparks debates on radio shows", "keeps fans talking"],
        "travel": ["draws curious crowds", "sails into port", "opens to visitors", "hosts tours all day", "gets a warm welcome"],
        "culture": ["draws long lines", "lights up downtown", "fills the venue", "gets strong local buzz", "brings people together"],
        "business": ["opens with fanfare", "hosts a packed launch", "fills the floor with guests", "draws coverage from papers", "sells out early"],
        "science": ["shows off new ideas", "welcomes students and families", "gets shared by educators", "draws local press", "debuts a demonstration"],
        "general": ["draws wide coverage", "gets local buzz", "brings crowds downtown", "lands headlines", "becomes a news favorite"],
    }
    modern_actions = {
        "tech": ["launches to the public", "rolls out broadly", "goes live for users", "debuts with a demo", "lands for early adopters"],
        "social_media": ["takes over feeds", "picks up momentum", "spreads across platforms", "sparks quick reactions", "draws big creator posts"],
        "gaming": ["fills servers fast", "tops player charts", "shakes up rankings", "packs esports streams", "sparks lore debates"],
        "music": ["tops playlists", "sells out dates", "goes viral on clips", "gets heavy play", "trends on radio"],
        "film_tv": ["wins fan polls", "drives binge nights", "gets quoted online", "lands strong reviews", "spawns memes"],
        "sports": classic_actions["sports"],
        "internet_culture": ["dominates forums", "spreads through memes", "fills comment sections", "inspires parody threads", "hits front pages"],
        "travel": classic_actions["travel"],
        "culture": classic_actions["culture"],
        "business": classic_actions["business"],
        "science": classic_actions["science"],
        "general": classic_actions["general"],
    }

    classic_reacts = {
        "music": ["covered by music press", "remembered in entertainment columns", "discussed by radio hosts", "featured in year-end lists", "mentioned by critics"],
        "film_tv": ["covered by entertainment reporters", "earning notable reviews", "picked up by major outlets", "featured in weekly roundups", "discussed on radio"],
        "sports": ["leading sports coverage that week", "picked up by national sports desks", "highlighted across sports news", "remembered as a season moment", "recapped on television"],
        "travel": ["covered by local media", "drawing attention from visitors", "making regional news", "mentioned in travel roundups", "written up in papers"],
        "culture": ["earning local headlines", "drawing coverage from arts press", "featured in cultural news", "noted in community reports", "written up in magazines"],
        "business": ["covered by business press", "reported across outlets", "featured in market news", "highlighted in industry reports", "covered by newspapers"],
        "science": ["reported by science desks", "covered in academic news", "earning coverage in journals", "featured in education reports", "discussed by researchers"],
        "general": ["covered by major outlets", "drawing wide news attention", "highlighted in reports that week", "remembered in news summaries", "discussed on radio shows"],
    }
    modern_reacts = {
        "tech": ["drawing wide press coverage", "earning headlines that week", "reported across tech outlets", "covered in major news"],
        "social_media": ["spreading quickly across platforms", "drawing widespread attention", "covered by news outlets", "noted in weekly recaps"],
        "gaming": ["earning coverage from major sites", "reported as a standout moment", "featured in gaming news", "covered widely that week"],
        "music": classic_reacts["music"],
        "film_tv": classic_reacts["film_tv"],
        "sports": classic_reacts["sports"],
        "internet_culture": ["covered by major blogs", "highlighted in online roundups", "becoming a noted moment online", "landing in weekly recaps"],
        "travel": classic_reacts["travel"],
        "culture": classic_reacts["culture"],
        "business": classic_reacts["business"],
        "science": classic_reacts["science"],
        "general": classic_reacts["general"],
    }

    domain_subjects = modern_subjects if use_modern else classic_subjects
    domain_actions = modern_actions if use_modern else classic_actions
    domain_reacts = modern_reacts if use_modern else classic_reacts

    locations = ["Chicago", "Seattle", "Toronto", "Melbourne", "Oslo", "Lisbon", "Seoul", "Austin", "Dublin", "Vancouver", "Cape Town", "Barcelona", "Reykjavik", "Mexico City", "Bangkok", "Helsinki", "Madrid", "Valencia", "Bilbao", "Geneva"]

    # ensure two different domains for variety
    def build_fake(exclude_domain: str | None = None) -> str:
        pick_pool = [d for d in base_domains if d != exclude_domain] or base_domains
        pick_domain = rng.choice(pick_pool)
        subject = rng.choice(domain_subjects.get(pick_domain, domain_subjects["general"]))
        action = rng.choice(domain_actions.get(pick_domain, domain_actions["general"]))
        reaction = rng.choice(domain_reacts.get(pick_domain, domain_reacts["general"]))
        place = rng.choice(locations)
        sentence = f"{subject} {action} in {place}, {reaction}."
        return _normalize_choice(sentence, min_len, max_len)

    candidates: list[str] = []
    attempts = 0
    while len(candidates) < 2 and attempts < 12:
        exclude = candidates and _infer_domain(candidates[-1]) or None
        cand = build_fake(exclude_domain=exclude)
        if min_len <= _wlen(cand) <= max_len and not _mostly_year_swap(cand, real_text):
            if all(not _mostly_year_swap(cand, c) for c in candidates):
                if not any(_too_similar(cand, c) for c in candidates):
                    candidates.append(cand)
        attempts += 1
    while len(candidates) < 2:
        candidates.append(build_fake())
    return candidates[0], candidates[1]

def _unsplash_for(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (small_image_url, attribution_text) for a search query, or (None, None) if not available.
    """
    if not UNSPLASH_ACCESS_KEY:
        return None, None

    # Build a gentle query from keywords
    q = " ".join(w for w in re.findall(r"[A-Za-z]{3,}", text.lower()) if w not in {
        "the","and","for","with","from","into","across","over","under","in","on","at","to",
        "of","by","an","a","is","are","was","were","becomes","formed","opens","announces",
        "first","second","third","city","national","world","major","new","old"
    })[:6] or ["history", "archive"]

    url = "https://api.unsplash.com/search/photos"
    try:
        j = _http_get_json(
            url,
            params={"query": q, "orientation": "squarish", "per_page": 1},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
        )
        if j.get("results"):
            r = j["results"][0]
            img = r["urls"]["small"]
            user = r["user"]
            name = user.get("name", "Unsplash photographer")
            username = user.get("username", "")
            attr = f"{name} — https://unsplash.com/@{username}"
            return img, attr
    except Exception:
        pass
    return None, None


def _pick_real_event() -> Tuple[str, str, Optional[int]]:
    today = _today_local_date()
    url = WIKI_URL.format(m=today.month, d=today.day)
    j = _http_get_json(url, headers={"User-Agent": "TimelineRoulette/1.1"})
    events = j.get("events", [])
    if not events:
        raise RuntimeError("No events returned from Wikipedia OTD")

    def _text(e: dict) -> str:
        return (e.get("text") or e.get("displaytitle") or "").strip()

    scored = []
    for e in events:
        t = _text(e)
        if not t or t.count(".") > 1:
            continue
        if _is_tragedy(t):
            continue
        year = e.get("year")
        try:
            year_val = int(year)
        except Exception:
            year_val = None
        domain = _infer_domain(t)
        scored.append((_score_for_youth(t, year_val), t, year_val, domain))

    scored.sort(key=lambda x: x[0], reverse=True)
    pop_domains = {"tech","social_media","gaming","music","film_tv","internet_culture","sports"}

    pick = None
    pick_year: Optional[int] = None
    for score, text, year, dom in scored:
        if dom in pop_domains and (year is None or year >= 1985):
            pick = text
            pick_year = year
            break
    if not pick:
        for score, text, year, dom in scored:
            if dom in pop_domains:
                pick = text
                pick_year = year
                break
    if not pick:
        for score, text, year, dom in scored:
            if year and year >= 1985:
                pick = text
                pick_year = year
                break
    if not pick and scored:
        pick = scored[0][1]
        pick_year = scored[0][2]
    if not pick:
        pick = "A notable event is recorded by historians."
        pick_year = None

    month_name = today.strftime("%B")
    return pick, month_name, pick_year

# --- replace _openai_image with this ---
def _openai_image(prompt: str) -> Optional[str]:
    """
    Generate a small image and return a browser-safe data URL.
    Returns None on any failure.
    """
    if not OPENAI_API_KEY:
        return None
    try:
        r = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": "256x256",
                "n": 1,
                # IMPORTANT: most responses are base64 now
                "response_format": "b64_json",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("data"):
            return None
        b64 = data["data"][0].get("b64_json")
        if not b64:
            # backward compatibility if a URL ever appears
            url = data["data"][0].get("url")
            return url or None
        # return embeddable data URL (works in <img src>)
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

# keep this constant
FALLBACK_ICON_URL = "/static/roulette/icons/star.svg"

def _recent_image_urls(limit: int = 5) -> set[str]:
    """
    Pull a small window of recent images so we don't repeat yesterday's picks.
    """
    try:
        rows = (
            db.session.query(
                TimelineRound.real_img_url,
                TimelineRound.fake1_img_url,
                TimelineRound.fake2_img_url,
            )
            .order_by(TimelineRound.round_date.desc())
            .limit(limit)
            .all()
        )
        urls: set[str] = set()
        for row in rows:
            for url in row:
                if url:
                    urls.add(url)
        return urls
    except Exception:
        return set()

# --- replace _image_for with this ---
def _image_for(text: str, used: set[str]) -> Tuple[str, str]:
    """
    Pick an image for `text`. Prefer Unsplash (fast), then OpenAI, then a visible local fallback.
    Deduplicate within a round using `used`.
    """
    # 1) Unsplash (prefer speed & reliability)
    if UNSPLASH_ACCESS_KEY:
        words = re.findall(r"[A-Za-z]{3,}", (text or "").lower())
        stop = {"the","and","for","with","from","into","across","over","under","in","on","at","to","of","by","an","a","is","are","was","were"}
        core = [w for w in words if w not in stop][:6] or ["history","archive","museum","document"]
        q = " ".join(core)
        try:
            j = _http_get_json(
                "https://api.unsplash.com/search/photos",
                params={"query": q, "orientation": "squarish", "per_page": 6, "content_filter": "high"},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            )
            for r in j.get("results", []):
                img = r.get("urls", {}).get("small")
                if img and img not in used:
                    used.add(img)
                    u = r.get("user", {}) or {}
                    attr = f"{u.get('name','Unsplash')} — https://unsplash.com/@{u.get('username','unsplash')}"
                    return img, attr
        except Exception:
            pass

    # 2) OpenAI image (now base64 data-url capable)
    if OPENAI_API_KEY:
        u = _openai_image(f"flat minimal illustration; muted colors; historical vibe; about: {text}")
        if u and u not in used:
            used.add(u)
            return u, "OpenAI generated"

    # 3) Guaranteed visible fallback (never blank)
    if FALLBACK_ICON_URL not in used:
        used.add(FALLBACK_ICON_URL)
    return FALLBACK_ICON_URL, ""

def ensure_today_round(force: int = 0) -> bool:
    """
    Generate (or update) today's Timeline Round.

    - force=0 : if exists, NO-OP (return True)
    - force=1 : UPDATE the existing row in place (keep guesses)
    - force=2 : DELETE today's guesses, then UPDATE the row in place
    """
    today = _today_local_date()

    # find today's round (if any)
    existing: TimelineRound | None = db.session.execute(
        select(TimelineRound).where(TimelineRound.round_date == today)
    ).scalar_one_or_none()

    if existing and force == 0:
        return True

    # 1) pick a real event + generate fakes, icons, images
    real_raw, month_name, real_year = _pick_real_event()
    real_soft = _soften_real_title(real_raw)
    domain = _infer_domain(real_soft)

    real_soft = _normalize_choice(real_soft, 20, 25)
    real_len = _wlen(real_soft)
    fake_min = max(20, real_len - 5)
    fake_max = min(25, real_len + 5)

    year_hint = real_year or _extract_year_hint(real_raw) or _extract_year_hint(real_soft)
    fake1, fake2 = _openai_fakes_from_real(real_soft, month_name, domain, fake_min, fake_max, year_hint=year_hint)

    def _normalize_target(text: str) -> str:
        normalized = _normalize_choice(text, fake_min, fake_max)
        # keep in domain with a light flair if it drifted
        if domain != "general" and _infer_domain(normalized) != domain:
            normalized = _normalize_choice(f"{normalized.rstrip('.')} {random.choice(_domain_flair(domain, year_hint))}.", fake_min, fake_max)
        if _is_tragedy(normalized):
            safe_seed = f"A {domain} highlight covered by news outlets"
            normalized = _normalize_choice(safe_seed, fake_min, fake_max)
        # guard domain mismatch for general by regenerating a neutral phrasing
        if domain == "general" and not _domain_ok(normalized, domain):
            fallback_domain = random.choice(["travel","culture","business","science","general"])
            fallback_subjects = {
                "travel": ["A historic ship visit", "A new ferry route", "A busy harbor festival", "A landmark train run", "A waterfront opening"],
                "culture": ["A museum opening", "A public art show", "A city parade", "A landmark restoration", "A film festival night"],
                "business": ["A flagship store launch", "A brand collaboration", "A startup milestone", "A major product drop", "A pop-up event"],
                "science": ["A planetarium reveal", "A space exhibit", "A green tech demo", "A science fair highlight", "A new lab opening"],
                "general": ["A city festival", "A major exhibit", "A national celebration", "A community milestone", "A public event"],
            }
            fallback_actions = {
                "travel": ["draws curious crowds", "sails into port", "opens to visitors", "hosts tours all day", "gets a warm welcome"],
                "culture": ["draws long lines", "lights up downtown", "fills the venue", "gets strong local buzz", "brings people together"],
                "business": ["opens with fanfare", "hosts a packed launch", "pulls a long line", "draws coverage from papers", "sells out early"],
                "science": ["shows off new ideas", "welcomes students and families", "gets shared by educators", "draws local press", "debuts a demo"],
                "general": ["draws wide coverage", "gets local buzz", "brings crowds downtown", "lands headlines", "becomes a news favorite"],
            }
            fallback_reacts = {
                "travel": ["covered by local news crews", "locals share stories", "radio mentions follow", "tourists talk about it"],
                "culture": ["covered by critics", "locals share stories", "press photos circulate", "people discuss it"],
                "business": ["customers line up early", "newspapers cover the drop", "people talk about purchases", "radio segments highlight it"],
                "science": ["teachers discuss it", "local news covers it", "families talk about it", "bulletins mention the debut"],
                "general": ["people talk about it for days", "local news covers it", "crowds show up", "it lands in recaps"],
            }
            fs = fallback_subjects[fallback_domain]
            fa = fallback_actions[fallback_domain]
            fr = fallback_reacts[fallback_domain]
            fallback_subject = random.choice(fs)
            fallback_action = random.choice(fa)
            fallback_react = random.choice(fr)
            neutral = f"{fallback_subject} {fallback_action}, and {fallback_react}."
            normalized = _normalize_choice(neutral, fake_min, fake_max)
        if year_hint and year_hint < 1995:
            normalized = re.sub(r"\b(online|internet|viral|streaming|platforms?)\b", "", normalized, flags=re.I)
            normalized = _normalize_choice(normalized, fake_min, fake_max)
        return normalized

    fake1 = _normalize_target(fake1)
    fake2 = _normalize_target(fake2)

    # enforce unique openings and avoid simple year swaps
    rng_norm = random.Random(int(time.time()) ^ (hash(real_soft) & 0xFFFFFFFF))
    def _openings(text: str) -> list[str]:
        return [w.lower() for w in _words(text)[:3]]

    if _mostly_year_swap(fake1, real_soft) or _openings(fake1) == _openings(real_soft):
        fake1 = _normalize_target(_remix_structure(fake1, rng_norm))
    if _mostly_year_swap(fake2, real_soft) or _openings(fake2) == _openings(real_soft):
        fake2 = _normalize_target(_remix_structure(fake2, rng_norm))
    if _mostly_year_swap(fake2, fake1) or _openings(fake2) == _openings(fake1):
        fake2 = _normalize_target(_remix_structure(fake2, rng_norm))

    real_icon = pick_icon_for_text(real_soft)
    f1_icon   = pick_icon_for_text(fake1)
    f2_icon   = pick_icon_for_text(fake2)

    # seed with recent picks to avoid showing yesterday's photos again
    used_urls: set[str] = _recent_image_urls(limit=5)
    real_img, real_attr = _image_for(real_soft, used_urls)
    f1_img,  f1_attr    = _image_for(fake1, used_urls)
    f2_img,  f2_attr    = _image_for(fake2, used_urls)
    real_img = real_img or FALLBACK_ICON_URL
    f1_img = f1_img or FALLBACK_ICON_URL
    f2_img = f2_img or FALLBACK_ICON_URL
    img_attr = real_attr or f1_attr or f2_attr or ""

    try:
        if existing:
            # force==2: clear guesses that reference today's round
            if force == 2:
                db.session.query(TimelineGuess).filter(
                    TimelineGuess.round_id == existing.id
                ).delete(synchronize_session=False)

            # UPDATE IN PLACE (keep same round id for FK)
            existing.real_title     = real_soft
            existing.real_source_url = f"https://en.wikipedia.org/wiki/{today.strftime('%B')}_{today.day}"
            existing.fake1_title    = fake1
            existing.fake2_title    = fake2
            existing.correct_index  = 0  # real is index 0; UI shuffles

            # icons / images / attr
            existing.real_icon      = real_icon
            existing.fake1_icon     = f1_icon
            existing.fake2_icon     = f2_icon

            existing.real_img_url   = real_img
            existing.fake1_img_url  = f1_img
            existing.fake2_img_url  = f2_img

            existing.real_img_attr  = img_attr
            # keep compatibility fields if they exist
            if hasattr(existing, "fake1_img_attr"): existing.fake1_img_attr = None
            if hasattr(existing, "fake2_img_attr"): existing.fake2_img_attr = None

        else:
            # CREATE if missing
            round_row = TimelineRound(
                round_date=today,
                real_title=real_soft,
                real_source_url=f"https://en.wikipedia.org/wiki/{today.strftime('%B')}_{today.day}",
                fake1_title=fake1,
                fake2_title=fake2,
                correct_index=0,
                real_icon=real_icon,
                fake1_icon=f1_icon,
                fake2_icon=f2_icon,
                real_img_url=real_img,
                fake1_img_url=f1_img,
                fake2_img_url=f2_img,
                real_img_attr=img_attr,
                fake1_img_attr=None,
                fake2_img_attr=None,
            )
            db.session.add(round_row)

        db.session.commit()
        return True

    except Exception as e:
        db.session.rollback()
        current_app.logger.exception(f"[roulette] ensure_today_round failed: {e}")
        return False
