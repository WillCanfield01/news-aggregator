# app/scripts/generate_timeline_round.py
from __future__ import annotations

import os
import re
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from flask import current_app
from app.roulette.models import TimelineRound, TimelineGuess
import pytz
import requests
from sqlalchemy import select

from app.extensions import db
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

QUOTE_LIBRARY: list[dict[str, str]] = [
    {"text": "The future depends on what you do today.", "author": "Mahatma Gandhi"},
    {"text": "Success is walking from failure to failure with no loss of enthusiasm.", "author": "Winston Churchill"},
    {"text": "We are what we repeatedly do. Excellence, then, is not an act but a habit.", "author": "Aristotle"},
    {"text": "It always seems impossible until it is done.", "author": "Nelson Mandela"},
    {"text": "The most difficult thing is the decision to act, the rest is merely tenacity.", "author": "Amelia Earhart"},
    {"text": "I am not afraid of storms, for I am learning how to sail my ship.", "author": "Louisa May Alcott"},
    {"text": "If you can dream it, you can do it.", "author": "Walt Disney"},
    {"text": "You miss one hundred percent of the shots you never take.", "author": "Wayne Gretzky"},
    {"text": "Do what you can, with what you have, where you are.", "author": "Theodore Roosevelt"},
    {"text": "Courage is grace under pressure.", "author": "Ernest Hemingway"},
    {"text": "In the middle of difficulty lies opportunity.", "author": "Albert Einstein"},
    {"text": "The secret of getting ahead is getting started.", "author": "Mark Twain"},
    {"text": "Dream big and dare to fail.", "author": "Norman Vaughan"},
    {"text": "The best way out is always through.", "author": "Robert Frost"},
    {"text": "Quality is not an act, it is a habit.", "author": "Aristotle"},
    {"text": "Well done is better than well said.", "author": "Benjamin Franklin"},
    {"text": "The future starts today, not tomorrow.", "author": "Pope John Paul II"},
    {"text": "Turn your wounds into wisdom.", "author": "Oprah Winfrey"},
    {"text": "Action is the foundational key to all success.", "author": "Pablo Picasso"},
    {"text": "Small deeds done are better than great deeds planned.", "author": "Peter Marshall"},
    {"text": "Success is the sum of small efforts, repeated day in and day out.", "author": "Robert Collier"},
    {"text": "The journey of a thousand miles begins with a single step.", "author": "Lao Tzu"},
    {"text": "Discipline is choosing between what you want now and what you want most.", "author": "Abraham Lincoln"},
    {"text": "Luck is a dividend of sweat. The more you sweat, the luckier you get.", "author": "Ray Kroc"},
    {"text": "Champions keep playing until they get it right.", "author": "Billie Jean King"},
    {"text": "I've failed over and over and that is why I succeed.", "author": "Michael Jordan"},
    {"text": "The harder the conflict, the more glorious the triumph.", "author": "Thomas Paine"},
    {"text": "Strive not to be a success, but rather to be of value.", "author": "Albert Einstein"},
    {"text": "Nothing will work unless you do.", "author": "Maya Angelou"},
    {"text": "Act as if what you do makes a difference. It does.", "author": "William James"},
    {"text": "What you do today can improve all your tomorrows.", "author": "Ralph Marston"},
    {"text": "Make each day your masterpiece.", "author": "John Wooden"},
    {"text": "Fortune favors the bold.", "author": "Virgil"},
    {"text": "The best revenge is massive success.", "author": "Frank Sinatra"},
    {"text": "Talent wins games, but teamwork wins championships.", "author": "Michael Jordan"},
    {"text": "If you want to lift yourself up, lift up someone else.", "author": "Booker T. Washington"},
    {"text": "Do not wait to strike till the iron is hot; make it hot by striking.", "author": "William Butler Yeats"},
    {"text": "Either you run the day or the day runs you.", "author": "Jim Rohn"},
    {"text": "If there is no struggle, there is no progress.", "author": "Frederick Douglass"},
    {"text": "Inspiration exists, but it has to find you working.", "author": "Pablo Picasso"},
    {"text": "Great things are done by a series of small things brought together.", "author": "Vincent van Gogh"},
    {"text": "Success usually comes to those who are too busy to be looking for it.", "author": "Henry David Thoreau"},
    {"text": "The only way around is through.", "author": "Robert Frost"},
    {"text": "You are never too old to set another goal or to dream a new dream.", "author": "C. S. Lewis"},
    {"text": "Pressure is a privilege.", "author": "Billie Jean King"},
    {"text": "Opportunities multiply as they are seized.", "author": "Sun Tzu"},
    {"text": "Energy and persistence conquer all things.", "author": "Benjamin Franklin"},
    {"text": "Keep your face always toward the sunshine, and shadows will fall behind you.", "author": "Walt Whitman"},
    {"text": "Progress lies not in enhancing what is, but in advancing toward what will be.", "author": "Khalil Gibran"},
    {"text": "Success is never accidental.", "author": "Jack Welch"},
    {"text": "Decide what you want, and then act as if it were impossible to fail.", "author": "Brian Tracy"},
    {"text": "Victory belongs to the most persevering.", "author": "Napoleon Bonaparte"},
    {"text": "Perfection is not attainable, but if we chase perfection we can catch excellence.", "author": "Vince Lombardi"},
    {"text": "You must do the thing you think you cannot do.", "author": "Eleanor Roosevelt"},
    {"text": "A goal without a plan is just a wish.", "author": "Antoine de Saint-Exupery"},
    {"text": "Never confuse a single defeat with a final defeat.", "author": "F. Scott Fitzgerald"},
    {"text": "The way to get started is to quit talking and begin doing.", "author": "Walt Disney"},
    {"text": "Big results require big ambitions.", "author": "Heraclitus"},
    {"text": "If you are working on something exciting that you really care about, you don't have to be pushed. The vision pulls you.", "author": "Steve Jobs"},
    {"text": "Hard work beats talent when talent doesn't work hard.", "author": "Kevin Durant"},
    {"text": "Don't wait. The time will never be just right.", "author": "Napoleon Hill"},
    {"text": "We become what we think about.", "author": "Earl Nightingale"},
    {"text": "Stay hungry, stay foolish.", "author": "Steve Jobs"},
    {"text": "Rise above the storm and you will find the sunshine.", "author": "Mario Fernandez"},
    {"text": "I am always doing that which I cannot do, in order that I may learn how to do it.", "author": "Pablo Picasso"},
    {"text": "The only place where success comes before work is in the dictionary.", "author": "Vidal Sassoon"},
    {"text": "Do not pray for an easy life, pray for the strength to endure a difficult one.", "author": "Bruce Lee"},
    {"text": "Motivation is what gets you started. Habit is what keeps you going.", "author": "Jim Ryun"},
    {"text": "Champions take chances when there are no chances.", "author": "Jack Dempsey"},
    {"text": "Vision without execution is just hallucination.", "author": "Henry Ford"},
    {"text": "Some people want it to happen, some wish it would happen, others make it happen.", "author": "Michael Jordan"},
    {"text": "The reward for work well done is the opportunity to do more.", "author": "Jonas Salk"},
    {"text": "I find that the harder I work, the more luck I seem to have.", "author": "Thomas Jefferson"},
    {"text": "Do not be afraid to give up the good to go for the great.", "author": "John D. Rockefeller"},
    {"text": "Never let the fear of striking out keep you from playing the game.", "author": "Babe Ruth"},
    {"text": "Success is liking yourself, liking what you do, and liking how you do it.", "author": "Maya Angelou"},
    {"text": "No pressure, no diamonds.", "author": "Thomas Carlyle"},
    {"text": "Fear is a reaction. Courage is a decision.", "author": "Winston Churchill"},
    {"text": "Luck is what happens when preparation meets opportunity.", "author": "Seneca"},
    {"text": "Do one thing every day that scares you.", "author": "Eleanor Roosevelt"},
    {"text": "You can waste your lives drawing lines. Or you can live your life crossing them.", "author": "Shonda Rhimes"},
    {"text": "The difference between the impossible and the possible lies in a person's determination.", "author": "Tommy Lasorda"},
    {"text": "Life shrinks or expands in proportion to one's courage.", "author": "Anais Nin"},
    {"text": "He who is not courageous enough to take risks will accomplish nothing in life.", "author": "Muhammad Ali"},
    {"text": "Believe you can and you're halfway there.", "author": "Theodore Roosevelt"},
    {"text": "You can't build a reputation on what you are going to do.", "author": "Henry Ford"},
    {"text": "Success is a science; if you have the conditions, you get the result.", "author": "Oscar Wilde"},
    {"text": "Give light and people will find the way.", "author": "Ella Baker"},
    {"text": "There are no shortcuts to any place worth going.", "author": "Beverly Sills"},
    {"text": "When you reach the end of your rope, tie a knot in it and hang on.", "author": "Franklin D. Roosevelt"},
    {"text": "Never allow a person to tell you no who doesn't have the power to say yes.", "author": "Eleanor Roosevelt"},
    {"text": "The only real mistake is the one from which we learn nothing.", "author": "Henry Ford"},
    {"text": "Patience, persistence and perspiration make an unbeatable combination for success.", "author": "Napoleon Hill"},
    {"text": "A river cuts through rock not because of its power, but because of its persistence.", "author": "James Watkins"},
    {"text": "Success is doing ordinary things extraordinarily well.", "author": "Jim Rohn"},
    {"text": "Great works are performed not by strength but by perseverance.", "author": "Samuel Johnson"},
    {"text": "Our greatest glory is not in never falling, but in rising every time we fall.", "author": "Confucius"},
    {"text": "Start where you are. Use what you have. Do what you can.", "author": "Arthur Ashe"},
    {"text": "We can do anything we want to if we stick to it long enough.", "author": "Helen Keller"},
    {"text": "Don't watch the clock; do what it does. Keep going.", "author": "Sam Levenson"},
    {"text": "Once you choose hope, anything's possible.", "author": "Christopher Reeve"},
    {"text": "Hold yourself responsible for a higher standard than anybody expects of you.", "author": "Henry Ward Beecher"},
    {"text": "If you fell down yesterday, stand up today.", "author": "H. G. Wells"},
    {"text": "Stay patient and trust your journey.", "author": "Kobe Bryant"},
    {"text": "When something is important enough, you do it even if the odds are not in your favor.", "author": "Elon Musk"},
    {"text": "The pain you feel today will be the strength you feel tomorrow.", "author": "Ritu Ghatourey"},
    {"text": "The best dreams happen when you are awake.", "author": "Cherie Gilderbloom"},
]

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

    # collapse very large counts like 12,345 or 2000000 -> "thousands"
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


def _headline_from_text(text: str, min_words: int = 6, max_words: int = 14) -> str:
    """
    Build a clean headline-style line without trailing punctuation.
    """
    words = _words(text)
    if not words:
        return ""
    trimmed = words[:max_words]
    if len(trimmed) < min_words and len(words) >= min_words:
        trimmed = words[:min_words]
    headline = " ".join(trimmed)
    headline = re.sub(r"[.?!,:;]+$", "", headline).strip()
    return _sentence_case(headline)


def _blurb_from_text(text: str, min_words: int = 12, max_words: int = 22) -> str:
    """
    Return a concise, single-sentence blurb with a soft clamp.
    """
    return _normalize_choice(text, min_words, max_words)

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
            "as fans lined up early",
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
            "during a talked-about trailer drop",
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
    Produce two plausible-but-false events similar in tone/length to the real item, with a single coherent action.
    """
    era_note = ""
    if year_hint:
        decade = (year_hint // 10) * 10
        era_note = f"Keep the era around the {decade}s."

    def _ask_openai() -> Optional[str]:
        sys_prompt = (
            "Write one believable 'On this day' news-style sentence (no bullet). "
            f"Length {min_len}-{max_len} words. One sentence only. No trailing fragments. "
            "No tragedies/violence. Avoid jargon. Use concrete nouns and a single clear action. "
            "Do not copy the real event. Do not prepend a year dash."
        )
        if era_note:
            sys_prompt += f" {era_note}"
        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Real event for context (do not copy): {real_text}\\n"
                        f"Month: {month_name}\\n"
                        f"Era: {year_hint or 'any'}"
                    ),
                },
            ],
            "temperature": 0.8,
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
        for _ in range(4):
            out = _ask_openai()
            if out and min_len <= _wlen(out) <= max_len and not _is_tragedy(out):
                if all(not _too_similar(out, f) for f in fakes) and not _too_similar(out, real_text):
                    fakes.append(out)
            if len(fakes) >= 2:
                return fakes[0], fakes[1]

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
        "music": ["a concert series", "a chart single", "a touring symphony", "a headline recital", "a festival night"],
        "film_tv": ["a film premiere", "a broadcast special", "a documentary debut", "a cinema release", "a celebrated screening"],
        "sports": ["a title match", "a record-setting game", "a heated rivalry", "a marathon milestone", "a playoff upset"],
        "travel": ["a historic ship visit", "a new ferry route", "a harbor festival", "a landmark train run", "a waterfront opening"],
        "culture": ["a museum opening", "a public art show", "a city parade", "a landmark restoration", "a film festival gala"],
        "business": ["a flagship store opening", "a brand collaboration", "a company milestone", "a major product unveiling", "a regional expo"],
        "science": ["a planetarium reveal", "a space exhibit", "a research announcement", "a science fair highlight", "a new laboratory wing"],
        "general": ["a national celebration", "a civic ceremony", "a community milestone", "a public event", "a major exhibit"],
    }
    modern_subjects = {
        "tech": ["an app launch", "a device update", "a smart feature rollout", "a cloud rollout", "a streaming upgrade"],
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
        "music": ["draws crowds", "lands on radio charts", "gets strong reviews", "sells out quickly"],
        "film_tv": ["earns solid reviews", "packs theaters", "airs to strong ratings", "wins a festival slot"],
        "sports": ["fills the arena", "leads sports pages", "is replayed on nightly news", "sparks debate shows"],
        "travel": ["welcomes visitors", "fills the docks", "hosts public tours", "draws curious locals"],
        "culture": ["draws long lines", "lights up downtown", "fills the venue", "gets strong local buzz"],
        "business": ["opens with fanfare", "fills the floor with guests", "draws coverage from papers", "sells out early"],
        "science": ["shows new work", "welcomes students", "draws local press", "features demonstrations"],
        "general": ["draws wide coverage", "gets local buzz", "brings crowds downtown", "lands headlines"],
    }
    modern_actions = {
        "tech": ["launches to the public", "rolls out broadly", "goes live for users", "debuts with demos"],
        "social_media": ["takes over feeds", "picks up momentum", "spreads across platforms", "sparks quick reactions", "draws big creator posts"],
        "gaming": ["fills servers fast", "tops player charts", "shakes up rankings", "packs esports streams", "sparks lore debates"],
        "music": ["tops playlists", "sells out dates", "goes viral on clips", "gets heavy play", "trends on radio"],
        "film_tv": ["wins fan polls", "drives binge nights", "gets quoted in recaps", "lands strong reviews", "spawns memes"],
        "sports": classic_actions["sports"],
        "internet_culture": ["dominates forums", "spreads through memes", "fills comment sections", "inspires parody threads", "hits front pages"],
        "travel": classic_actions["travel"],
        "culture": classic_actions["culture"],
        "business": classic_actions["business"],
        "science": classic_actions["science"],
        "general": classic_actions["general"],
    }

    classic_reacts = {
        "music": ["covered by critics", "remembered in entertainment pages", "noted on radio"],
        "film_tv": ["covered by entertainment reporters", "picked up by major outlets", "featured in reviews"],
        "sports": ["leading sports coverage", "highlighted across sports news", "remembered that season"],
        "travel": ["covered by local media", "making regional news", "mentioned in travel roundups"],
        "culture": ["earning local headlines", "covered by cultural desks", "noted in community reports"],
        "business": ["covered by business press", "reported across outlets", "featured in market news"],
        "science": ["reported by science desks", "covered in academic news", "earning coverage in journals"],
        "general": ["covered by major outlets", "drawing wide news attention", "remembered in summaries"],
    }
    modern_reacts = {
        "tech": ["drawing wide press coverage", "earning headlines that week", "reported across tech outlets"],
        "social_media": ["spreading quickly across platforms", "drawing widespread attention", "covered by news outlets", "noted in weekly recaps"],
        "gaming": ["earning coverage from major sites", "reported as a standout moment", "featured in gaming news", "covered widely that week"],
        "music": classic_reacts["music"],
        "film_tv": classic_reacts["film_tv"],
        "sports": classic_reacts["sports"],
        "internet_culture": ["covered by major blogs", "highlighted in weekend roundups", "landing in weekly recaps"],
        "travel": classic_reacts["travel"],
        "culture": classic_reacts["culture"],
        "business": classic_reacts["business"],
        "science": classic_reacts["science"],
        "general": classic_reacts["general"],
    }

    domain_subjects = modern_subjects if use_modern else classic_subjects
    domain_actions = modern_actions if use_modern else classic_actions
    domain_reacts = modern_reacts if use_modern else classic_reacts

    locations = ["Chicago", "Seattle", "Toronto", "Melbourne", "Oslo", "Lisbon", "Seoul", "Austin", "Dublin", "Vancouver", "Cape Town", "Barcelona", "Reykjavik", "Mexico City", "Bangkok", "Helsinki", "Madrid", "Valencia", "Bilbao", "Geneva", "Monaco", "Vienna", "Prague"]

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
    while len(candidates) < 2 and attempts < 16:
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

# Optional image generation helper retained for compatibility (not user-facing).
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
                "prompt": f"highly realistic wide-angle photograph, full scene in frame, natural lighting, no text or logos, {prompt}",
                "size": "1024x1024",
                "n": 1,
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
            url = data["data"][0].get("url")
            return url or None
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None

# keep this constant
FALLBACK_ICON_URL = "/static/roulette/icons/star.svg"
# 1x1 transparent pixel to avoid broken imgs if static path fails
PLACEHOLDER_IMG = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="

FALLBACK_IMAGE_MAP: Dict[str, list[str]] = {
    "politics": ["politics-01.svg", "politics-02.svg"],
    "science": ["science-01.svg", "science-02.svg"],
    "sports": ["sports-01.svg", "sports-02.svg"],
    "business": ["business-01.svg", "business-02.svg"],
    "culture": ["culture-01.svg", "culture-02.svg"],
    "weather": ["weather-01.svg", "weather-02.svg"],
    "general": ["general-01.svg", "general-02.svg"],
}

LOCATION_KEYWORDS = {
    "new york": "New York",
    "los angeles": "Los Angeles",
    "washington": "Washington",
    "boston": "Boston",
    "chicago": "Chicago",
    "miami": "Miami",
    "seattle": "Seattle",
    "denver": "Denver",
    "dallas": "Dallas",
    "houston": "Houston",
    "austin": "Austin",
    "san francisco": "San Francisco",
    "london": "London",
    "paris": "Paris",
    "berlin": "Berlin",
    "rome": "Rome",
    "madrid": "Madrid",
    "lisbon": "Lisbon",
    "oslo": "Oslo",
    "stockholm": "Stockholm",
    "vienna": "Vienna",
    "prague": "Prague",
    "toronto": "Toronto",
    "vancouver": "Vancouver",
    "mexico city": "Mexico City",
    "sydney": "Sydney",
    "melbourne": "Melbourne",
    "tokyo": "Tokyo",
    "seoul": "Seoul",
    "singapore": "Singapore",
    "cape town": "Cape Town",
}

EVENT_TYPE_KEYWORDS = [
    ("election", "politics", ["election", "vote", "ballot", "referendum", "runoff", "polls"]),
    ("treaty", "politics", ["treaty", "accord", "agreement", "pact", "summit"]),
    ("protest", "politics", ["protest", "march", "rally", "demonstration", "sit-in"]),
    ("launch", "science", ["launch", "rocket", "spacecraft", "capsule", "satellite", "mission"]),
    ("museum opening", "culture", ["museum", "exhibit", "gallery", "exhibition", "art show"]),
    ("concert", "culture", ["concert", "festival", "tour", "performance", "orchestra", "stage"]),
    ("tournament", "sports", ["championship", "final", "tournament", "cup", "league", "marathon", "race", "match", "season opener"]),
    ("stadium", "sports", ["stadium", "arena", "ballpark", "court", "pitch"]),
    ("acquisition", "business", ["acquisition", "merger", "ipo", "earnings", "revenue", "shares", "stock", "deal"]),
    ("storm", "weather", ["hurricane", "storm", "typhoon", "cyclone", "tropical", "flood", "wildfire", "blizzard", "tornado"]),
    ("earthquake", "weather", ["earthquake", "aftershock", "quake"]),
    ("research", "science", ["research", "experiment", "lab", "telescope", "observatory"]),
]

CATEGORY_KEYWORDS = {
    "weather": ["storm", "hurricane", "flood", "wildfire", "earthquake", "blaze", "tornado", "typhoon", "cyclone", "mudslide"],
    "politics": ["election", "parliament", "president", "senate", "treaty", "accord", "government", "minister", "campaign"],
    "sports": ["match", "tournament", "league", "cup", "final", "olympics", "stadium", "championship", "season opener"],
    "science": ["launch", "space", "satellite", "telescope", "mission", "laboratory", "research", "experiment", "observatory"],
    "business": ["market", "merger", "acquisition", "ipo", "earnings", "deal", "shares", "company", "startup"],
    "culture": ["museum", "gallery", "concert", "festival", "exhibit", "premiere", "theater", "parade", "exhibition"],
}


def _category_for_domain(domain: str) -> str:
    mapping = {
        "tech": "business",
        "social_media": "business",
        "gaming": "culture",
        "music": "culture",
        "film_tv": "culture",
        "sports": "sports",
        "travel": "culture",
        "culture": "culture",
        "business": "business",
        "science": "science",
        "internet_culture": "culture",
        "general": "general",
    }
    return mapping.get(domain, "general")


def _category_for_text(text: str, domain: str) -> str:
    lc = (text or "").lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(w in lc for w in words):
            return cat
    return _category_for_domain(domain)


def _guess_location(text: str) -> Optional[str]:
    lc = (text or "").lower()
    for key, val in LOCATION_KEYWORDS.items():
        if key in lc:
            return val
    m = re.search(r"\b(?:in|at|near|outside|across)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})", text or "")
    if m:
        guess = m.group(1).strip()
        if len(guess.split()) <= 3:
            return guess
    return None


def _guess_entity(text: str) -> Optional[str]:
    stop = {"the", "a", "an", "this", "that", "its", "their"}
    months = {m.lower() for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", text or "")
    for cand in candidates:
        lc = cand.lower()
        if lc in stop or lc in months:
            continue
        if lc in LOCATION_KEYWORDS:
            continue
        if len(cand) < 3:
            continue
        return cand.strip()
    return None


def _guess_event_type(text: str) -> Tuple[Optional[str], Optional[str]]:
    lc = (text or "").lower()
    for label, category, keywords in EVENT_TYPE_KEYWORDS:
        if any(w in lc for w in keywords):
            return label, category
    return None, None


def _build_image_queries(choice: dict) -> list[str]:
    entity = choice.get("entity")
    location = choice.get("location")
    event_type = choice.get("event_type")
    category = choice.get("category")
    year = choice.get("year")
    queries: list[str] = []

    combo = " ".join(p for p in [entity, location, event_type] if p)
    if combo:
        queries.append(f"{combo} {year}" if year else combo)
    if location and event_type:
        queries.append(f"{location} {event_type}")
    if event_type:
        queries.append(f"{event_type} news photo")
    if category:
        queries.append(f"{category} event photo")
    # drop empties / duplicates
    seen: set[str] = set()
    final: list[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        final.append(q)
    return final


def _validate_image_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    if url.startswith("data:image"):
        return url
    if url.startswith("/static/"):
        try:
            static_root = Path(current_app.static_folder)
        except Exception:
            static_root = None
        rel = url.lstrip("/")
        if static_root and (static_root / rel).exists():
            return url
        return None
    if not url.startswith("https://"):
        return None
    try:
        r = requests.head(url, allow_redirects=True, timeout=6)
        ctype = r.headers.get("Content-Type", "").lower()
        if r.status_code == 200 and "image" in ctype:
            return url
    except Exception:
        return None
    return None


def _recent_image_urls(limit: int = 15) -> set[str]:
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
                valid = _validate_image_url(url)
                if valid:
                    urls.add(valid)
        return urls
    except Exception:
        return set()


def _fallback_image_for_category(category: str, used: set[str], rng: random.Random) -> str:
    names = FALLBACK_IMAGE_MAP.get(category) or FALLBACK_IMAGE_MAP["general"]
    pool = list(names)
    rng.shuffle(pool)
    for name in pool:
        url = f"/static/roulette/fallbacks/{name}"
        if url in used:
            continue
        used.add(url)
        return url
    used.add(FALLBACK_ICON_URL)
    return FALLBACK_ICON_URL


def _search_unsplash(query: str, rng: random.Random, used: set[str]) -> Tuple[Optional[str], Optional[str]]:
    if not UNSPLASH_ACCESS_KEY or not query:
        return None, None
    page = 1 + rng.randint(0, 2)
    try:
        j = _http_get_json(
            "https://api.unsplash.com/search/photos",
            params={
                "query": query,
                "orientation": "landscape",
                "per_page": 8,
                "page": page,
                "content_filter": "high",
                "order_by": "relevant",
            },
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
        )
        for r in j.get("results", []):
            urls = r.get("urls", {}) or {}
            img = urls.get("regular") or urls.get("full") or urls.get("small")
            if not img or img in used:
                continue
            valid = _validate_image_url(img)
            if not valid or valid in used:
                continue
            used.add(valid)
            u = r.get("user", {}) or {}
            attr = f"{u.get('name','Unsplash')} - https://unsplash.com/@{u.get('username','unsplash')}"
            return valid, attr
    except Exception:
        return None, None
    return None, None


def _generate_decoy_image(choice: dict, rng: random.Random, used: set[str], seed: int | None = None) -> Tuple[Optional[str], Optional[str]]:
    prompt_seed = seed or rng.randint(10_000, 99_999)
    if OPENAI_API_KEY:
        prompt = (
            "documentary style photograph, natural lighting, wide frame, no text or graphics, "
            f"news photo of: {choice.get('text','')}. seed {prompt_seed}"
        )
        cand = _openai_image(prompt)
        valid = _validate_image_url(cand)
        if valid and valid not in used:
            used.add(valid)
            return valid, None
    found, attr = _search_unsplash(choice.get("text", ""), rng, used)
    if found:
        return found, attr
    return None, None


def pick_image_for_choice(choice: dict, rng: random.Random, used: set[str], existing_url: str | None = None, mode: str = "real", seed: int | None = None) -> Tuple[str, str, Optional[str]]:
    """
    Pick an image for a choice, honoring existing URLs when valid.
    mode: "real" uses Unsplash-first; "decoy" prefers generated decoys.
    Returns (url, source_type, attribution).
    """
    existing_valid = _validate_image_url(existing_url)
    if existing_valid and existing_valid not in used:
        used.add(existing_valid)
        return existing_valid, "cached", None

    if mode == "real":
        queries = _build_image_queries(choice)
        for q in queries:
            found, attr = _search_unsplash(q, rng, used)
            if found:
                return found, "search", attr
        # one last pass with the raw text
        found, attr = _search_unsplash(choice.get("text", ""), rng, used)
        if found:
            return found, "search", attr
        fallback = _fallback_image_for_category(choice.get("category") or "general", used, rng)
        return fallback, "fallback", None

    # decoys: try generated, then unsplash, then fallback
    url, attr = _generate_decoy_image(choice, rng, used, seed)
    if url:
        return url, "generated", attr
    found, attr = _search_unsplash(choice.get("text", ""), rng, used)
    if found:
        return found, "search", attr
    fallback = _fallback_image_for_category(choice.get("category") or "general", used, rng)
    return fallback, "fallback", None

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

    real_soft = _normalize_choice(real_soft, 14, 22)
    real_len = _wlen(real_soft)
    fake_min = max(14, real_len - 3)
    fake_max = min(22, real_len + 3)

    year_hint = real_year or _extract_year_hint(real_raw) or _extract_year_hint(real_soft)
    fake1, fake2 = _openai_fakes_from_real(real_soft, month_name, domain, fake_min, fake_max, year_hint=year_hint)

    def _normalize_target(text: str) -> str:
        normalized = _normalize_choice(text, fake_min, fake_max)
        # keep in domain with a light flair if it drifted
        if domain != "general" and _infer_domain(normalized) != domain:
            if random.random() < 0.4:
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

    def _choice_meta(text: str) -> dict:
        ev_type, ev_cat = _guess_event_type(text)
        domain_guess = _infer_domain(text)
        category = ev_cat or _category_for_text(text, domain_guess)
        return {
            "text": text,
            "domain": domain_guess,
            "category": category,
            "event_type": ev_type or category,
            "location": _guess_location(text),
            "entity": _guess_entity(text),
            "year": year_hint,
        }

    real_meta = _choice_meta(real_soft)
    fake1_meta = _choice_meta(fake1)
    fake2_meta = _choice_meta(fake2)

    # seed with recent picks to avoid showing yesterday's photos again
    used_urls: set[str] = _recent_image_urls(limit=8)

    existing_imgs = {
        "real": _validate_image_url(getattr(existing, "real_img_url", None)) if existing else None,
        "fake1": _validate_image_url(getattr(existing, "fake1_img_url", None)) if existing else None,
        "fake2": _validate_image_url(getattr(existing, "fake2_img_url", None)) if existing else None,
    }
    existing_attrs = {
        "real": getattr(existing, "real_img_attr", None) if existing else None,
        "fake1": getattr(existing, "fake1_img_attr", None) if existing else None,
        "fake2": getattr(existing, "fake2_img_attr", None) if existing else None,
    }

    # avoid duplicates when reusing cached images
    already: set[str] = set()
    for key in ("real", "fake1", "fake2"):
        url = existing_imgs[key]
        if url and url not in already:
            already.add(url)
        else:
            existing_imgs[key] = None
    used_urls |= already

    seed_base = int(datetime.now(TZ).strftime("%Y%m%d"))
    rng = random.Random(seed_base)
    real_img, real_src, real_attr_new = pick_image_for_choice(real_meta, rng, used_urls, existing_imgs["real"], mode="real")
    f1_img, _, f1_attr_new = pick_image_for_choice(fake1_meta, rng, used_urls, existing_imgs["fake1"], mode="decoy", seed=seed_base + 11)
    f2_img, _, f2_attr_new = pick_image_for_choice(fake2_meta, rng, used_urls, existing_imgs["fake2"], mode="decoy", seed=seed_base + 19)

    # If the real pick fell back, keep all three on consistent fallbacks for credibility
    if real_src == "fallback":
        used_urls = {real_img}
        f1_img = _fallback_image_for_category(fake1_meta["category"], used_urls, rng)
        f2_img = _fallback_image_for_category(fake2_meta["category"], used_urls, rng)
        f1_attr_new = None
        f2_attr_new = None

    real_img = real_img or FALLBACK_ICON_URL or PLACEHOLDER_IMG
    f1_img = f1_img or FALLBACK_ICON_URL or PLACEHOLDER_IMG
    f2_img = f2_img or FALLBACK_ICON_URL or PLACEHOLDER_IMG
    real_attr = real_attr_new or existing_attrs["real"] or ""
    f1_attr = f1_attr_new or existing_attrs["fake1"] or ""
    f2_attr = f2_attr_new or existing_attrs["fake2"] or ""
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
            if hasattr(existing, "fake1_img_attr"): existing.fake1_img_attr = f1_attr or None
            if hasattr(existing, "fake2_img_attr"): existing.fake2_img_attr = f2_attr or None

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
                fake1_img_attr=f1_attr or None,
                fake2_img_attr=f2_attr or None,
            )
            db.session.add(round_row)

        db.session.commit()
        return True

    except Exception as e:
        db.session.rollback()
        current_app.logger.exception(f"[roulette] ensure_today_round failed: {e}")
        return False
