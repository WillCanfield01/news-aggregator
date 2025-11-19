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

# --- adjacency helpers -------------------------------------------------

ENTITY_SWAPS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\bGoogle\b"), ["Microsoft", "Apple", "Yahoo", "Amazon"]),
    (re.compile(r"\bMicrosoft\b"), ["Google", "Apple", "IBM"]),
    (re.compile(r"\bApple\b"), ["Samsung", "Google", "Sony"]),
    (re.compile(r"\bNASA\b"), ["ESA", "Roscosmos", "JAXA"]),
    (re.compile(r"\bSpace Shuttle Columbia\b"), ["Space Shuttle Discovery", "Space Shuttle Atlantis", "Space Shuttle Endeavour"]),
    (re.compile(r"\bSpace Shuttle Discovery\b"), ["Space Shuttle Columbia", "Space Shuttle Atlantis", "Space Shuttle Endeavour"]),
    (re.compile(r"\bSpace Shuttle Atlantis\b"), ["Space Shuttle Columbia", "Space Shuttle Discovery", "Space Shuttle Endeavour"]),
    (re.compile(r"\bSpace Shuttle Endeavour\b"), ["Space Shuttle Columbia", "Space Shuttle Discovery", "Space Shuttle Atlantis"]),
    (re.compile(r"\bLos Angeles\b"), ["New York", "Seattle", "San Francisco", "Chicago"]),
    (re.compile(r"\bNew York\b"), ["Los Angeles", "Chicago", "Boston"]),
]

def _mutate_event_like(real_text: str, rng: random.Random) -> str:
    """
    Create a fake that stays close to the real event:
    - same general topic and sentence shape
    - tweak org / vehicle / place / mission number / year
    """
    s = real_text.strip()

    # 1) tweak an organization / place / vehicle if present
    for pattern, alts in ENTITY_SWAPS:
        if pattern.search(s):
            replacement = rng.choice(alts)
            s = pattern.sub(replacement, s)
            break

    # 2) tweak STS mission numbers like STS-87 -> STS-89
    m = re.search(r"\bSTS-(\d+)\b", s)
    if m:
        try:
            n = int(m.group(1))
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_n = max(1, n + delta)
            s = s[:m.start(1)] + str(new_n) + s[m.end(1):]
        except ValueError:
            pass

    # 3) tweak the year if present
    y = _extract_year(s)
    if y:
        delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])
        new_y = max(1900, y + delta)
        s = re.sub(r"\b(19|20)\d{2}\b", str(new_y), s, count=1)

    # clean up spacing and punctuation
    s = s.strip()
    if not s.endswith("."):
        s += "."

    return s

def _fallback_adjacent_fakes(real_text: str, rng: random.Random) -> Tuple[str, str]:
    """
    Deterministic fallback when OpenAI is unavailable.
    - For space-style events: one shuttle variation + one related satellite/mission.
    - For others: first fake is a mutation of the real, second fake is a mutation of the first.
    """
    lc = real_text.lower()

    # Space / shuttle style events
    if "space shuttle" in lc or "sts-" in lc or "nasa" in lc or "orbit" in lc:
        year = _extract_year(real_text) or rng.randint(1981, 2011)

        # variation of the original
        fake1 = _mutate_event_like(real_text, rng)

        # adjacent but different scenario (satellite or different kind of mission)
        mission_num = rng.randint(60, 95)
        fake2 = (
            f"A communications satellite is deployed during NASA mission STS-{mission_num} "
            f"to expand global coverage in {year}."
        )

        # small safety: avoid accidental duplicates
        if fake2.strip().lower() == fake1.strip().lower():
            mission_num += 2
            fake2 = (
                f"A joint NASA–ESA mission STS-{mission_num} conducts microgravity experiments "
                f"on board a laboratory module in {year}."
            )
        return fake1, fake2

    # Generic path: chain two different mutations
    fake1 = _mutate_event_like(real_text, rng)
    base_for_second = fake1
    fake2 = _mutate_event_like(base_for_second, rng)
    tries = 0
    while (
        fake2.strip().lower() in {
            fake1.strip().lower(),
            real_text.strip().lower(),
        } and tries < 5
    ):
        fake2 = _mutate_event_like(base_for_second, rng)
        tries += 1

    return fake1, fake2

def _openai_fakes_from_real(real_text: str, month_name: str) -> Tuple[str, str]:
    """
    Produce two fakes that feel like close siblings of the real event.
    They should:
      - stay in the same topic family (space mission, band, game, law, etc.)
      - have similar length and style
      - be different from each other in subject / wording
    """
    target_words = max(8, _wlen(real_text))
    domain = _infer_domain(real_text)
    real_year = _extract_year(real_text)

    # ------------------------------------------
    # 1) OpenAI path (preferred)
    # ------------------------------------------
    if OPENAI_API_KEY:
        sys_prompt = (
            "You write short 'On This Day' history facts for a daily guessing game.\n"
            "You will be given ONE real event.\n"
            "Create TWO different events that are plausible but false.\n"
            "They must look like they belong on the same list as the real event.\n"
            "\n"
            "Rules for BOTH fakes:\n"
            "- Keep the same general topic family as the original "
            "(space missions with space missions, sports with sports, tech with tech, etc.).\n"
            "- Stay roughly in the same era. Years should be believable for that topic.\n"
            "- Each fake should be within ±20% of the original's word count and under 30 words.\n"
            "- Each fake must include at least one proper noun and one 4-digit year.\n"
            "- Avoid wars, disasters, or mass-casualty events.\n"
            "- Use simple, readable language. No corporate-speak like 'targeting younger users'.\n"
            "\n"
            "Differences between fake1 and fake2:\n"
            "- fake1 should feel like a variation of the original: similar kind of event, "
            "  but change mission/name/place/year.\n"
            "- fake2 should be a related but clearly different scenario in the same topic family "
            "  (for example a follow-up mission, a similar achievement by another organization, "
            "  or a parallel milestone).\n"
            "- fake1 and fake2 must NOT describe the exact same main subject or mission.\n"
            "\n"
            "Return STRICT JSON only: {\"fake1\": \"...\", \"fake2\": \"...\"}"
        )

        user_content = (
            f"Original event: {real_text}\n"
            f"Month: {month_name}\n"
            f"Guessed topic/domain: {domain}\n"
            f"Approximate year (if present): {real_year if real_year else 'unknown'}\n"
            f"Target word count: around {target_words} words."
        )

        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
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
            if (
                f1
                and f2
                and f1.strip().lower() != f2.strip().lower()
            ):
                return f1, f2
        except Exception:
            time.sleep(0.3)  # fall through to deterministic fallback

    # ------------------------------------------
    # 2) Deterministic fallback: adjacent fakes
    # ------------------------------------------
    rng = random.Random(
        int(datetime.now(TZ).strftime("%Y%m%d")) ^ (hash(real_text) & 0xFFFFFFFF)
    )
    return _fallback_adjacent_fakes(real_text, rng)

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

    # take the top few and randomize so you can re-roll the real event
    top = scored[:20]
    random.shuffle(top)

    picked_text: Optional[str] = None

    # first try high scoring, post-1980, single sentence, reasonable length
    for e in top:
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

    # fallback to any clean item inside the same randomized top set
    if not picked_text:
        for e in top:
            t = _text(e)
            if 40 <= len(t) <= 200 and t.count(".") <= 1:
                picked_text = t
                break

    # final fallback: original rough filter over all events
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
                headers={"Authorization": "Client-ID " + UNSPLASH_ACCESS_KEY},
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
