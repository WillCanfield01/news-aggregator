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

# --- replace _openai_fakes_from_real with this stronger version ---
def _openai_fakes_from_real(real_text: str, month_name: str) -> Tuple[str, str]:
    """
    Produce two plausible-but-false events similar in tone/length to the real item.
    """
    # ---------- 1) Try OpenAI ----------
    if OPENAI_API_KEY:
        sys_prompt = (
            "You write concise, believable 'On This Day' almanac entries.\n"
            "Given ONE real entry, return TWO different **plausible but false** entries that:\n"
            f"• Occur in {month_name} (any year),\n"
            "• Match the tone and length (within ±15%) of the real entry,\n"
            "• Each include at least ONE proper noun (org/place/person) and ONE 4-digit year,\n"
            "• Sound specific (a concrete action or outcome),\n"
            "• DO NOT copy exact entities from the real entry,\n"
            "• No meta-text. No hedging words like 'reportedly' or 'allegedly'.\n"
            'Return STRICT JSON only: {\"fake1\":\"...\",\"fake2\":\"...\"}'
        )
        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"real_entry={real_text}"},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        for _ in range(2):
            try:
                r = requests.post(
                    OAI_URL,
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
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
                if f1 and f2 and f1.lower() != real_text.lower() and f2.lower() != real_text.lower():
                    return f1, f2
            except Exception:
                time.sleep(0.5)

    # ---------- 2) Deterministic fallback (no API / API failed) ----------
    # make fakes carry named entities + years to feel 'real'
    rng = random.Random(int(datetime.now(TZ).strftime("%Y%m%d")) ^ (hash(real_text) & 0xFFFFFFFF))
    regions = ["France", "Canada", "Brazil", "Japan", "India", "Italy", "Kenya", "Australia", "Spain", "Norway"]
    cities  = ["Marseille", "Quebec City", "São Paulo", "Osaka", "Pune", "Milan", "Nairobi", "Perth", "Valencia", "Bergen"]
    bodies  = ["National Assembly", "Supreme Court", "Railway Commission", "Postal Service", "Maritime Authority",
               "Central Bank", "Broadcasting Corporation", "University Senate", "City Council", "Museum Board"]
    verbs   = ["authorizes", "ratifies", "suspends", "standardizes", "opens", "formally adopts", "repeals", "announces"]
    objects = ["a nationwide licensing scheme", "new safety codes", "a public broadcaster charter",
               "provincial tax rules", "a national archive program", "interstate tariffs", "coastal fishing limits"]
    years   = list(range(1860, 2022))

    def one():
        region = rng.choice(regions)
        city   = rng.choice(cities)
        body   = rng.choice(bodies)
        verb   = rng.choice(verbs)
        obj    = rng.choice(objects)
        year   = rng.choice(years)
        # target similar length
        target_len = max(60, min(180, int(len(real_text)*rng.uniform(0.85, 1.15))))
        s = f"{body} in {city}, {region}, {verb} {obj} in {year}."
        # pad lightly with a motive/impact clause (but stay concise)
        tails = [
            "to streamline regional policy.",
            "after months of debate.",
            "citing budget pressures.",
            "following a contested vote.",
            "to align with international norms."
        ]
        if len(s) < target_len and rng.random() < 0.8:
            s = s[:-1] + " " + rng.choice(tails)
        return s

    f1 = one()
    f2 = one()
    if f2.lower() == f1.lower():
        f2 = one()
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
    j = _http_get_json(url, headers={"User-Agent": "TimelineRoulette/1.0"})
    events = j.get("events", [])
    if not events:
        raise RuntimeError("No events returned from Wikipedia OTD")

    # Pick a mid-specificity item (avoid extremely niche or multisentence)
    random.shuffle(events)
    candidates = []
    for e in events:
        t = e.get("text") or e.get("displaytitle") or ""
        if not t:
            continue
        if 40 <= len(t) <= 180 and t.count(".") <= 1:
            candidates.append(t.strip())
    if not candidates:
        candidates = [e.get("text", "").strip() for e in events if e.get("text")]

    real = (candidates[0] if candidates else "A notable event is recorded by historians.")
    # also return month name for hinting
    month_name = today.strftime("%B")
    return real, month_name

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