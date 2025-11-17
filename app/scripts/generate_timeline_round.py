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

def _extract_year(text: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", text or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None

# Domains and nostalgia signals for younger / TikTok crowd
POP_KEYWORDS = {
    "social_media": ["tiktok","instagram","facebook","twitter","x.com","snapchat","youtube","reddit","twitch","vine","myspace"],
    "gaming": ["nintendo","playstation","xbox","pokemon","minecraft","fortnite","roblox","steam","esports","blizzard","sony"],
    "music": ["spotify","itunes","mtv","grammy","billboard","taylor swift","drake","bts","k-pop","eminem","nirvana"],
    "film_tv": ["netflix","disney","marvel","star wars","hbo","pixar","oscars","hulu","prime video","anime"],
    "tech": ["iphone","android","apple","google","microsoft","ai","openai","tesla","spacex","nvidia","samsung","internet","www"],
    "sports": ["nba","nfl","mlb","nhl","fifa","world cup","olympics","super bowl","lakers","yankees","patriots"],
    "internet_culture": ["meme","viral","hashtag","emoji","stream","podcast","blog","wiki","open source","linux","browser"],
}

NEGATIVE_TERMS = [
    "killed", "killing", "dead", "dies", "death", "fatal",
    "massacre", "genocide", "famine",
    "bomb", "bombing", "explosion", "blast",
    "attack", "assault", "raid", "airstrike", "suicide bomber",
    "war", "battle", "invasion", "occupation", "uprising", "coup",
    "shooting", "gunman", "terrorist", "hostage",
    "crash", "collides", "collision", "derailment",
    "earthquake", "hurricane", "typhoon", "tsunami", "flood", "wildfire",
    "eruption", "eruptions", "erupts", "volcano", "volcanic",
]

def _infer_domain(text: str) -> str:
    """Rough guess: tech / gaming / sports / etc."""
    if not text:
        return "general"
    lc = text.lower()

    if any(k in lc for k in ("tiktok","instagram","youtube","snapchat","twitter","x.com","reddit","twitch")):
        return "social_media"
    if any(k in lc for k in ("nintendo","playstation","xbox","pokemon","fortnite","roblox","minecraft","steam","esports")):
        return "gaming"
    if any(k in lc for k in ("spotify","itunes","mtv","billboard","grammy","taylor swift","drake","bts","k-pop","eminem","nirvana")):
        return "music"
    if any(k in lc for k in ("netflix","disney","marvel","star wars","hbo","anime","pixar","oscars","prime video","hulu")):
        return "film_tv"
    if any(k in lc for k in ("iphone","android","apple","google","microsoft","ai","openai","tesla","spacex","nvidia","samsung")):
        return "tech"
    if any(k in lc for k in ("nba","nfl","mlb","nhl","fifa","world cup","olympics","super bowl","lakers","yankees")):
        return "sports"
    if any(k in lc for k in ("meme","viral","hashtag","emoji","podcast","blog","wiki","browser","internet")):
        return "internet_culture"
    return "general"

def _is_tragedy(text: str) -> bool:
    lc = (text or "").lower()
    return any(w in lc for w in NEGATIVE_TERMS)

def _score_for_youth(event_obj: dict) -> float:
    """
    Score a Wikipedia event for 'TikTok generation nostalgia'.

    High score:
      - tech / gaming / social / music / film / sports / internet culture
      - year roughly 1985–2018 (extra sweet spot 1995–2012)
      - short and readable
      - includes modern pop keywords
    Strong penalty:
      - wars, disasters, mass deaths
      - ancient or medieval history
    """
    t = (event_obj.get("text") or event_obj.get("displaytitle") or "").strip()
    if not t:
        return -1e9
    if _is_tragedy(t):
        return -1e9

    lc = t.lower()
    score = 0.0

    domain = _infer_domain(t)
    pop_domains = {"tech","social_media","gaming","music","film_tv","internet_culture","sports"}
    if domain in pop_domains:
        score += 4.0
    elif domain != "general":
        score += 0.5

    # bonus if specific pop keywords appear
    for cat, words in POP_KEYWORDS.items():
        if any(w in lc for w in words):
            score += 1.5

    # nostalgia year curve
    year = event_obj.get("year")
    try:
        y = int(year)
        if 1995 <= y <= 2012:
            score += 4.0
        elif 1985 <= y <= 1994:
            score += 3.0
        elif 2013 <= y <= 2018:
            score += 2.0
        elif 1960 <= y < 1985:
            score += 1.0
        elif y < 1950:
            score -= 2.5
    except Exception:
        pass

    # concision
    L = len(t)
    if 60 <= L <= 160:
        score += 1.0
    elif L > 220:
        score -= 0.5

    return score

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

def _openai_fakes_from_real(real_text: str, month_name: str) -> Tuple[str, str]:
    """
    Produce two fakes that feel like simple, pop-culture 'On This Day' facts.
    They should:
      - match the era and general domain of the real event
      - feel relevant to tech, games, social, music, film, sports, or internet
      - be short and easy to read
    """
    target_len = max(10, _wlen(real_text))
    domain = _infer_domain(real_text)
    real_year = _extract_year(real_text)

    # ------------------------------------------
    # 1) OpenAI path
    # ------------------------------------------
    if OPENAI_API_KEY:
        domain_hint = {
            "tech": "tech milestones, gadgets, internet companies",
            "social_media": "social platforms, creator culture, viral features",
            "gaming": "console launches, game studios, esports moments",
            "music": "artists, albums, charts, streaming moments",
            "film_tv": "movies, TV premieres, streaming originals",
            "sports": "major league moments and sports culture",
            "internet_culture": "memes, online communities, viral internet history",
        }.get(domain, "modern pop culture and internet history")

        sys_prompt = (
            "You write short, punchy pop-culture 'On This Day' facts for a daily trivia game "
            "aimed at Gen Z and Millennials.\n"
            f"Month: {month_name}.\n"
            f"Domain: {domain_hint}.\n"
            "Given ONE real event, invent TWO different events that are plausible but false.\n"
            "Rules:\n"
            "- Each event must be under 26 words.\n"
            "- Each must include at least one recognizable name (brand, platform, artist, team, show) and one 4 digit year.\n"
            "- Style should feel like a fun history headline, not a government report.\n"
            "- No long bureaucratic phrases, no councils, no committees, no regulations.\n"
            "- No tragedies, crashes, wars, disasters, or mass deaths.\n"
            "- Keep language simple and readable.\n"
            "- Do not reuse the same main entities as the real event.\n"
            'Return STRICT JSON only: {"fake1":"...","fake2":"..."}'
        )

        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Real event: {real_text}"},
            ],
            "temperature": 0.8,
            "response_format": {"type": "json_object"},
        }

        try:
            r = requests.post(
                OAI_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            try:
                obj = requests.utils.json.loads(content)
            except Exception:
                obj = requests.utils.json.loads(content.strip("` \n"))
            f1 = (obj.get("fake1") or "").strip()
            f2 = (obj.get("fake2") or "").strip()
            if f1 and f2:
                return f1, f2
        except Exception:
            # fall through to deterministic fallback
            time.sleep(0.3)

    # ------------------------------------------
    # 2) Deterministic pop-culture fallback
    # ------------------------------------------
    rng = random.Random(
        int(datetime.now(TZ).strftime("%Y%m%d")) ^ (hash(real_text) & 0xFFFFFFFF)
    )

    brands = [
        "Netflix", "Disney", "Spotify", "Apple", "Google", "Microsoft",
        "Nintendo", "PlayStation", "Xbox", "TikTok", "Instagram", "YouTube",
        "Twitch", "Reddit", "Epic Games", "HBO", "MTV"
    ]

    verbs = [
        "launches", "debuts", "announces", "tests", "rolls out",
        "brings back", "adds", "premieres", "updates"
    ]

    features = [
        "a new streaming feature",
        "a retro themed interface",
        "a limited event for fans",
        "a cross platform update",
        "a creator focused tool",
        "a live concert special",
        "a classic game collection",
        "a surprise season of a hit show",
        "a short form video mode",
    ]

    places = [
        "in Los Angeles", "in Tokyo", "in London", "in New York",
        "in Seoul", "in Toronto", "in Sydney", "in Berlin"
    ]

    def pick_year() -> int:
        if real_year:
            lo = max(1985, real_year - 5)
            hi = min(2022, real_year + 5)
            if lo <= hi:
                return rng.randint(lo, hi)
        return rng.randint(1990, 2020)

    def build_fake() -> str:
        year = pick_year()
        s = f"{rng.choice(brands)} {rng.choice(verbs)} {rng.choice(features)} {rng.choice(places)} in {year}."
        # keep roughly near real length but still short
        while _wlen(s) > max(target_len + 4, 22):
            parts = s.split()
            s = " ".join(parts[:-1])
            if not s.endswith("."):
                s += "."
        return s

    f1 = build_fake()
    f2 = build_fake()
    while f2 == f1:
        f2 = build_fake()

    return f1, f2

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

def _pick_real_event() -> Tuple[str, str]:
    today = _today_local_date()
    url = WIKI_URL.format(m=today.month, d=today.day)
    j = _http_get_json(url, headers={"User-Agent": "TimelineRoulette/1.1"})
    events = j.get("events", [])
    if not events:
        raise RuntimeError("No events returned from Wikipedia OTD")

    def _text(e: dict) -> str:
        return (e.get("text") or e.get("displaytitle") or "").strip()

    # filter out tragedies first
    pool = [e for e in events if _text(e) and not _is_tragedy(_text(e))]
    if not pool:
        pool = [e for e in events if _text(e)]

    # score for youth appeal
    scored = sorted(pool, key=_score_for_youth, reverse=True)

    picked_text: Optional[str] = None

    # first try high scoring, post-1980, single sentence, reasonable length
    for e in scored[:20]:
        t = _text(e)
        year = e.get("year")
        try:
            y = int(year)
        except Exception:
            y = None
        if y is not None and y < 1980:
            continue
        if 40 <= len(t) <= 200 and t.count(".") <= 1:
            picked_text = t
            break

    # fallback to any high-scoring event that reads cleanly
    if not picked_text:
        for e in scored[:20]:
            t = _text(e)
            if 40 <= len(t) <= 200 and t.count(".") <= 1:
                picked_text = t
                break

    # final fallback: original rough filter
    if not picked_text:
        random.shuffle(events)
        for e in events:
            t = _text(e)
            if 40 <= len(t) <= 200 and t.count(".") <= 1:
                picked_text = t
                break

    if not picked_text:
        picked_text = "A pop-culture or tech milestone is recorded."

    month_name = today.strftime("%B")
    return picked_text, month_name

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
            timeout=30,
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
    real_raw, month_name = _pick_real_event()
    real_soft = _soften_real_title(real_raw)
    fake1, fake2 = _openai_fakes_from_real(real_soft, month_name)

    real_icon = pick_icon_for_text(real_soft)
    f1_icon   = pick_icon_for_text(fake1)
    f2_icon   = pick_icon_for_text(fake2)

    used_urls: set[str] = set()
    real_img, real_attr = _image_for(real_soft, used_urls)
    f1_img,  f1_attr    = _image_for(fake1, used_urls)
    f2_img,  f2_attr    = _image_for(fake2, used_urls)
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