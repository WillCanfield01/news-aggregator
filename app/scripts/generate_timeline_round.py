# app/scripts/generate_timeline_round.py
from __future__ import annotations

import os
import re
import random
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

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
    Make the real title a hair less “hyper-specific” so it doesn't scream 'I'm the real one'.
    We keep meaning but remove dead giveaways: big counts, hyper-exact years, long proper-noun runs.
    """
    # (1) collapse large numbers
    t = re.sub(r"\b(\d{4,}|\d{1,3}(,\d{3})+)\b", "thousands", title)

    # (2) replace exact years like 'in 1862' with 'in the 19th century' when possible
    t = re.sub(r"\b(in|In)\s+(1[5-9]\d{2}|20\d{2})\b", "in that era", t)

    # (3) if we have very long proper-noun spans (3+ caps words), soften
    def _soften_caps_span(m):
        return "a major institution"

    t = re.sub(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})\b", _soften_caps_span, t)

    # normalize double spaces after replacements
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _openai_fakes_from_real(real_text: str, month_name: str) -> Tuple[str, str]:
    """
    Try OpenAI; if unavailable or fails, synthesize two DISTINCT fake entries.
    Deterministic per-day so users see stable choices.
    """
    # ---------- 1) Try OpenAI ----------
    if OPENAI_API_KEY:
        sys_prompt = (
            "You are a careful historian writing daily almanac entries. "
            "Given a real event, produce two false-but-plausible events that could appear in a daily timeline. "
            "They MUST:\n"
            "• be on the SAME month/day window (any year OK),\n"
            "• match the real entry’s tone and approximate length,\n"
            "• avoid copying exact names in the real entry (similar *type* is fine),\n"
            "• avoid meta commentary, never say they are fake.\n"
            "Return STRICT JSON only: {\"fake1\": \"...\", \"fake2\": \"...\"}."
        )
        user_payload = {"real_event": real_text, "month_hint": month_name}
        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"{user_payload}"},
            ],
            "temperature": 0.8,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(2):
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
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    obj = requests.utils.json.loads(content)
                except Exception:
                    obj = requests.utils.json.loads(content.strip("` \n"))
                f1 = (obj.get("fake1") or "").strip()
                f2 = (obj.get("fake2") or "").strip()
                # quick sanity: non-empty and not obviously duplicates of real
                if f1 and f2 and f1.lower() != real_text.lower() and f2.lower() != real_text.lower():
                    return f1, f2
            except Exception:
                time.sleep(0.6)

    # ---------- 2) Deterministic fallback (no API / API failed) ----------
    base = re.sub(r"[.!?]\s*$", "", real_text or "").strip()
    # Token set to avoid copying long real tokens
    real_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", base)}
    # Seed by day so results are stable for the calendar day
    _seed = int(datetime.now(TZ).strftime("%Y%m%d"))
    rng = random.Random(_seed ^ (hash(base) & 0xFFFFFFFF))

    # Word banks (neutral, historical tone)
    actors = [
        "regional leaders", "city officials", "trade guilds", "local magistrates",
        "delegates", "civic groups", "scholars", "town elders", "envoys", "council members"
    ]
    actions = [
        "announce", "convene", "ratify", "broker", "publish", "propose",
        "draft", "endorse", "negotiate", "finalize"
    ]
    objects = [
        "a provisional accord", "a charter reform", "a trade memorandum",
        "an interim council", "a policy revision", "a public statement",
        "new ordinances", "a compact to reduce unrest", "a joint declaration",
        "a limited treaty"
    ]
    contexts = [
        "after tense negotiations", "amid growing unease", "following weeks of debate",
        "to calm unrest in key districts", "citing fiscal pressures",
        "to resolve a longstanding dispute", "in response to petitions",
        "as observers report a narrow vote", "after private deliberations",
        "to standardize local practices"
    ]

    def _unique_pick(pool: list[str]) -> str:
        rng.shuffle(pool)
        for item in pool:
            if not any(tok in real_tokens for tok in re.findall(r"[A-Za-z]{4,}", item)):
                return item
        return pool[0]

    def _compose() -> str:
        a = _unique_pick(actors[:])
        v = _unique_pick(actions[:])
        o = _unique_pick(objects[:])
        c = _unique_pick(contexts[:])
        # Two varied sentence shapes for diversity
        if rng.random() < 0.5:
            s = f"{a[0].upper() + a[1:]} {v} {o} {c}."
        else:
            s = f"After {c}, {a} {v} {o}."
        # Light clean & period
        s = re.sub(r"\s{2,}", " ", s).strip()
        if not s.endswith("."):
            s += "."
        return s

    def _len_ok(target: str, cand: str) -> bool:
        if not target or not cand:
            return False
        t, c = len(target), len(cand)
        return (0.55 * t) <= c <= (1.55 * t)

    # Draw two distinct candidates
    f1, f2 = "", ""
    for _ in range(6):
        cand = _compose()
        if _len_ok(base, cand) and cand.lower() != base.lower():
            if not f1:
                f1 = cand
            elif cand.lower() != f1.lower():
                f2 = cand
                break
    if not f1:
        f1 = "Regional leaders announce a provisional accord after tense negotiations."
    if not f2:
        f2 = "Delegates unveil a charter reform to resolve a longstanding dispute."

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


def ensure_today_round(force: bool = False) -> bool:
    """Creates today's round if missing (or if force==True). Returns True when a round exists."""
    today = _today_local_date()
    existing: TimelineRound | None = db.session.execute(
        select(TimelineRound).where(TimelineRound.round_date == today)
    ).scalar_one_or_none()

    if existing and not force:
        return True

    # If we’re forcing, try to delete existing safely
    if existing and force:
        # foreign keys might exist; let DB cascade if configured
        db.session.delete(existing)
        db.session.commit()
        existing = None

    # 1) pick a real event
    real_raw, month_name = _pick_real_event()
    real_soft = _soften_real_title(real_raw)

    # 2) get two balanced fakes based on the softened real
    fake1, fake2 = _openai_fakes_from_real(real_soft, month_name)

    # 3) icons + unsplash per card
    real_icon = pick_icon_for_text(real_soft)
    f1_icon = pick_icon_for_text(fake1)
    f2_icon = pick_icon_for_text(fake2)

    real_img, real_attr = _unsplash_for(real_soft)
    f1_img, f1_attr = _unsplash_for(fake1)
    f2_img, f2_attr = _unsplash_for(fake2)

    # prefer at least one attribution string for footer; keep your previous behavior
    img_attr = real_attr or f1_attr or f2_attr

    # 4) persist
    round_row = TimelineRound(
        round_date=today,
        real_title=real_soft,
        real_source_url=f"https://en.wikipedia.org/wiki/{_today_local_date().strftime('%B')}_{_today_local_date().day}",
        fake1_title=fake1,
        fake2_title=fake2,
        correct_index=0,  # real = 0 before shuffling in route
        real_icon=real_icon,
        fake1_icon=f1_icon,
        fake2_icon=f2_icon,
        real_img_url=real_img,
        fake1_img_url=f1_img,
        fake2_img_url=f2_img,
        real_img_attr=img_attr,   # store one attribution string
        fake1_img_attr=None,
        fake2_img_attr=None,
    )
    db.session.add(round_row)
    db.session.commit()
    return True
