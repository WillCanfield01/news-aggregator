# app/scripts/generate_timeline_round.py
from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import pytz
import requests
from flask import current_app

from app.extensions import db
from app.roulette.models import TimelineRound, TimelineGuess  # type: ignore


# --------------------------------------------------------------------------------------
# Local time helpers
# --------------------------------------------------------------------------------------

_LOCAL_TZ = pytz.timezone(os.getenv("LOCAL_TZ", "America/Denver"))

def _now_local_dt() -> datetime:
    return datetime.now(_LOCAL_TZ)

def _now_local_date() -> date:
    return _now_local_dt().date()


# --------------------------------------------------------------------------------------
# Icon picking (works with whatever SVGs you have)
# --------------------------------------------------------------------------------------

ICON_DIR = Path(__file__).resolve().parents[1] / "static" / "roulette" / "icons"

# Common safe fallbacks (first one that actually exists wins)
_FALLBACK_CANDIDATES = [
    "circle-dot.svg", "dot.svg", "circle.svg",
    "image.svg", "star.svg", "asterisk.svg",
    "compass.svg", "history.svg",
]

def _first_existing(names: List[str]) -> Optional[str]:
    for n in names:
        if (ICON_DIR / n).exists():
            return n
    return None

def _fallback_icon_name() -> str:
    f = _first_existing(_FALLBACK_CANDIDATES)
    if f:
        return f
    any_svg = next((p.name for p in ICON_DIR.glob("*.svg")), None)
    return any_svg or "circle.svg"

_FALLBACK_ICON = _fallback_icon_name()

_KEYWORD_ICON_CANDIDATES = {
    # transport
    "train": ["train.svg", "tram.svg", "subway.svg", "locomotive.svg"],
    "rail":  ["train.svg", "tram.svg", "railway.svg"],
    "ship": ["ship.svg", "boat.svg", "ferry.svg", "anchor.svg"],
    "bridge": ["bridge.svg", "landmark.svg", "building.svg"],
    "plane": ["plane.svg", "airplane.svg", "jet.svg"],
    "flight": ["plane.svg", "airplane.svg"],
    "zeppelin": ["airship.svg", "blimp.svg", "balloon.svg", "plane.svg"],

    # politics / conflict
    "war": ["swords.svg", "shield.svg", "flag.svg"],
    "treaty": ["scroll.svg", "file-text.svg", "file.svg"],
    "republic": ["flag.svg", "government.svg"],
    "election": ["ballot.svg", "vote.svg", "check-square.svg"],
    "parliament": ["ballot.svg", "flag.svg", "building.svg"],
    "king": ["crown.svg", "flag.svg"],
    "queen": ["crown.svg", "flag.svg"],

    # science / tech
    "scientist": ["flask.svg", "beaker.svg", "test-tube.svg", "lab.svg"],
    "experiment": ["flask.svg", "lightbulb.svg", "atom.svg"],
    "physics": ["atom.svg", "lightbulb.svg", "flask.svg"],
    "chemistry": ["flask.svg", "beaker.svg", "test-tube.svg"],
    "computer": ["cpu.svg", "monitor.svg", "laptop.svg"],
    "internet": ["globe.svg", "network.svg"],
    "satellite": ["satellite.svg", "antenna.svg"],
    "discovery": ["lightbulb.svg", "search.svg", "compass.svg"],

    # culture / sports
    "music": ["music.svg", "music-2.svg", "headphones.svg"],
    "museum": ["landmark.svg", "building.svg"],
    "library": ["book.svg", "library.svg"],
    "book": ["book.svg"],
    "chess": ["chess.svg", "chess-knight.svg"],
    "game": ["joystick.svg", "gamepad.svg", "dice.svg"],
    "medal": ["medal.svg", "trophy.svg"],
}

_CATEGORY_CANDIDATES = {
    "transport": ["compass.svg", "map.svg", "navigation.svg", "road.svg"],
    "politics":  ["flag.svg", "building.svg", "scales.svg"],
    "science":   ["lightbulb.svg", "atom.svg", "flask.svg"],
    "culture":   ["star.svg", "music.svg", "trophy.svg"],
}

_TRANSPORT = {"train","rail","railway","tram","ship","navy","sail","plane","aviation","flight","airship","zeppelin","bridge","canal"}
_POLITICS  = {"war","battle","treaty","republic","election","constitution","parliament","king","queen","empire"}
_SCIENCE   = {"scientist","laboratory","experiment","physics","chemistry","computer","internet","satellite","telegraph","metal","alloy","discovery","invention"}
_CULTURE   = {"music","symphony","band","museum","library","book","chess","game","olympic","tournament"}

_STOP = {
    "the","a","an","of","and","for","to","in","on","at","with","from","into","during","across","over","under",
    "new","old","first","second","third","year","day","city","state","country","national","world","begins","is","are",
    "was","were","becomes","formed","opens","announces","occurs","founded","founds","established","establishes",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")

def _words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]

def pick_icon_for_text(text: str) -> str:
    # exact keyword
    for w in _words(text):
        cand = _KEYWORD_ICON_CANDIDATES.get(w)
        if cand:
            n = _first_existing(cand)
            if n:
                return n

    ws = set(_words(text))
    if ws & _TRANSPORT:
        n = _first_existing(_CATEGORY_CANDIDATES["transport"])
        if n: return n
    if ws & _POLITICS:
        n = _first_existing(_CATEGORY_CANDIDATES["politics"])
        if n: return n
    if ws & _SCIENCE:
        n = _first_existing(_CATEGORY_CANDIDATES["science"])
        if n: return n
    if ws & _CULTURE:
        n = _first_existing(_CATEGORY_CANDIDATES["culture"])
        if n: return n

    return _FALLBACK_ICON


# --------------------------------------------------------------------------------------
# Wikipedia + local fallback
# --------------------------------------------------------------------------------------

WIKI_URL_TMPL = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{mm}/{dd}"

# small emergency cache for when Wikipedia 403s or returns empty;
# add more if you want — format: "MM-DD": [(title, url), ...]
_LOCAL_EVENTS = {
    "10-07": [
        ("The German Democratic Republic is formed.", "https://en.wikipedia.org/wiki/East_Germany"),
        ("The Granite Railway begins operations in the U.S.", "https://en.wikipedia.org/wiki/Granite_Railway"),
    ],
    "10-13": [
        ("The present church at Westminster Abbey is consecrated.", "https://en.wikipedia.org/wiki/Westminster_Abbey"),
        ("B'nai B'rith is founded in New York City.", "https://en.wikipedia.org/wiki/B%27nai_B%27rith"),
    ],
}

def _wiki_headers() -> dict:
    # Use a real UA to avoid 403; set a contact if you want
    contact = os.getenv("WIKI_CONTACT", "hello@therealroundup.com")
    return {
        "User-Agent": f"therealroundup.com TimelineRoulette (+{contact})",
        "Accept": "application/json",
    }

def fetch_onthisday_events(target: date) -> List[Tuple[str, str]]:
    mm = f"{target.month:02d}"
    dd = f"{target.day:02d}"
    url = WIKI_URL_TMPL.format(mm=mm, dd=dd)

    try:
        r = requests.get(url, headers=_wiki_headers(), timeout=12)
        r.raise_for_status()
        data = r.json()
        out: List[Tuple[str, str]] = []
        for ev in data.get("events", []):
            # prefer the first page title+uri if present
            pages = ev.get("pages") or []
            if pages:
                # the "displaytitle" often contains the useful label (short)
                t = pages[0].get("displaytitle") or pages[0].get("normalizedtitle") or pages[0].get("title")
                page_url = pages[0].get("content_urls", {}).get("desktop", {}).get("page") or pages[0].get("extract", "")
                if t and page_url:
                    # Combine with the excerpted text of the event itself if available
                    text = ev.get("text") or t
                    # Make a clean sentence-ish title
                    out.append((text.strip().rstrip("." ) + ".", page_url))
            elif ev.get("text"):
                out.append((ev["text"].strip().rstrip(".") + ".", "https://en.wikipedia.org/wiki/Main_Page"))
        return out
    except Exception:
        pass  # fall through to local cache

    key = f"{mm}-{dd}"
    return _LOCAL_EVENTS.get(key, []).copy()


# --------------------------------------------------------------------------------------
# Text shaping: soften “real” and generate believable fakes
# --------------------------------------------------------------------------------------

# Replace proper nouns / specifics with generic phrases so the real line isn't obviously “the specific one”.
_PN_MULTI   = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z']+)+)\b")
_CITY_HEADS = re.compile(r"\b(New|Los|San)\s[A-Z][a-z]+\b")
_LANDMARK   = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z']+)*)\s(Abbey|Cathedral|Bridge|Library|Palace)\b")
_ORG        = re.compile(r"\b([A-Z][A-Za-z'&\.\-]+(?:\s[A-Z][A-Za-z'&\.\-]+)*)\b")

def soften_real_title(s: str) -> str:
    out = s
    out = _LANDMARK.sub(r"a well-known \2", out)
    out = _CITY_HEADS.sub("a major city", out)
    # collapse “Some Proper Noun” → “a long-standing civic organization” (only if followed by verbs like “is founded”, etc.)
    out = re.sub(rf"{_PN_MULTI.pattern}\s(is|was|were|is founded|is formed|is consecrated|is created)",
                 r"a long-standing civic organization \2", out)
    # general proper-noun wipe (gentle)
    out = _PN_MULTI.sub("a prominent institution", out)
    # trim doubled spaces / punctuation
    out = re.sub(r"\s+", " ", out).strip()
    if not out.endswith("."):
        out += "."
    return out

# Fake generators with tone similar to real
_FAKE_TEMPLATES = [
    "a scientific society is formed, drawing public attention.",
    "a new bridge is inaugurated, noted in reports.",
    "a major library opens its doors to the public.",
    "a civic treaty is announced after lengthy talks.",
    "a regional council is established to coordinate efforts.",
    "a historical archive is unveiled to researchers.",
    "a prominent exhibition opens, attracting crowds.",
    "a coastal fortification is completed.",
    "a postal route is expanded between major towns.",
    "a local academy is chartered for higher learning."
]

def _shuffle_take(k: int, items: List[str]) -> List[str]:
    pool = items[:]
    random.shuffle(pool)
    return pool[:k]

def generate_fake_titles(real_display: str) -> Tuple[str, str]:
    # lightly adapt verbs to match tense of real_display (past vs present)
    past = any(x in real_display.lower() for x in ["was ", "were ", "is ", "are ", "is founded", "is formed"])
    choices = _shuffle_take(6, _FAKE_TEMPLATES)
    if past:
        # keep as-is; templates are present-ish narrative which reads fine either way
        pass
    fake1, fake2 = choices[0], choices[1]
    return fake1, fake2


# --------------------------------------------------------------------------------------
# Core: ensure_today_round()
# --------------------------------------------------------------------------------------

@dataclass
class RoundData:
    real_title: str
    real_display: str
    real_url: str
    fake1: str
    fake2: str
    real_icon: str
    fake1_icon: str
    fake2_icon: str
    img_attr: Optional[str] = None  # reserved for Unsplash credit if you re-enable

def _pick_real_event_for_date(d: date) -> Tuple[str, str]:
    events = fetch_onthisday_events(d)
    if not events:
        # maximal last-resort line
        return ("A civic milestone is observed.", "https://en.wikipedia.org/wiki/Main_Page")
    # Filter out over-long or weird ones, pick a reasonable sample
    events = [(t, u) for (t, u) in events if 40 <= len(t) <= 180]
    if not events:
        t, u = random.choice(fetch_onthisday_events(d))
        return (t, u)
    return random.choice(events)

def _build_round_for(d: date) -> RoundData:
    real_raw, real_url = _pick_real_event_for_date(d)
    real_display = soften_real_title(real_raw)

    fake1, fake2 = generate_fake_titles(real_display)

    # icons
    real_icon  = pick_icon_for_text(real_display)
    fake1_icon = pick_icon_for_text(fake1)
    fake2_icon = pick_icon_for_text(fake2)

    return RoundData(
        real_title=real_raw.rstrip(".") + ".",
        real_display=real_display,
        real_url=real_url,
        fake1=fake1,
        fake2=fake2,
        real_icon=real_icon or _FALLBACK_ICON,
        fake1_icon=fake1_icon or _FALLBACK_ICON,
        fake2_icon=fake2_icon or _FALLBACK_ICON,
        img_attr=None,
    )

def seed_schema_if_missing() -> None:
    """
    Safety helper: in case your DB was created without these tables/columns.
    Call from a shell once: with app.app_context(): seed_schema_if_missing()
    """
    # We rely on SQLAlchemy's metadata and db.create_all() (which you already call in app/app.py).
    # This function just exists for parity with earlier raw SQL attempts and is no-op now.
    pass

def ensure_today_round(force: bool = False) -> TimelineRound:
    """
    Creates (or replaces when force=True) today's TimelineRound.
    - Safe against FK errors: deletes guesses first if replacing.
    - Softens real title so it doesn't look obviously specific.
    - Stores icon file names (your route helper maps to URL+fallback).
    """
    today = _now_local_date()

    existing = TimelineRound.query.filter_by(round_date=today).first()
    if existing and not force:
        return existing

    if existing and force:
        # Delete guesses first to satisfy FK, then delete round
        TimelineGuess.query.filter_by(round_id=existing.id).delete(synchronize_session=False)
        db.session.delete(existing)
        db.session.commit()

    data = _build_round_for(today)

    # correct_index is 0 (we treat real as the canonical slot; UI shuffles)
    # Store the softened display text for the real *in the visible text* by setting real_title to softened?
    # We’ll keep the exact real_title for logging/source, but save softened variant into DB’s real_title
    # so the UI shows the softened form. If you prefer both, add a new column.
    round_row = TimelineRound(
        round_date=today,
        real_title=data.real_display,      # **softened** for display
        real_source_url=data.real_url,
        fake1_title=data.fake1,
        fake2_title=data.fake2,
        correct_index=0,
        real_icon=data.real_icon,
        fake1_icon=data.fake1_icon,
        fake2_icon=data.fake2_icon,
        # keep attribution fields ready even if we don’t fetch Unsplash here
        real_img_attr=data.img_attr,
        fake1_img_attr=data.img_attr,
        fake2_img_attr=data.img_attr,
    )
    db.session.add(round_row)
    db.session.commit()
    return round_row


# --------------------------------------------------------------------------------------
# CLI / Render shell helpers
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Local manual test (must be run under `flask shell` or create_app() context)
    # from app.app import create_app
    # app = create_app()
    # with app.app_context():
    #     ensure_today_round(force=True)
    print("This module provides ensure_today_round(force=False). Run it inside app context.")
