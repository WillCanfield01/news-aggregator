# app/scripts/generate_timeline_round.py
import os
import random
import time
import datetime as dt
import requests
import pytz

from app.extensions import db
from app.roulette.models import TimelineRound

# ---- Config ----
TZ = os.getenv("TR_TZ", "America/Denver")  # adjust if you want UTC or another tz
USER_AGENT = os.getenv(
    "TR_UA",
    "TimelineRoulette/1.0 (+https://therealroundup.com; admin@therealroundup.com)",
)

WIKI_REST = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{m}/{d}"

# ---- Utils ----
def _now_local_date() -> dt.date:
    tz = pytz.timezone(TZ)
    return dt.datetime.now(tz).date()

def _http_get_json(url: str, max_retries: int = 3, timeout: int = 20) -> dict:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 403:
                # If we still ever hit 403, break early with a clearer message
                r.raise_for_status()
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # simple backoff
            time.sleep(min(1.0 * attempt, 3.0))
    raise last_err

def pick_real_event(today: dt.date) -> tuple[str, str]:
    """
    Returns (short_text, canonical_url) for a real event on this month/day.
    Picks a short, punchy entry if possible.
    """
    data = _http_get_json(WIKI_REST.format(m=today.month, d=today.day))
    events = [e for e in data.get("events", []) if e.get("text") and e.get("pages")]
    if not events:
        # Extremely rare; fallback to a safe default to keep the game alive
        return ("On this day: a notable historical event occurred.", "https://en.wikipedia.org/wiki/Main_Page")

    # prefer shorter texts for snappy UI
    events.sort(key=lambda e: len(e.get("text", "")))
    chosen = random.choice(events[:20]) if len(events) >= 20 else random.choice(events)
    title = chosen["text"][:300]
    url = chosen["pages"][0]["content_urls"]["desktop"]["page"]
    return title, url

def generate_fakes_with_llm(real_title: str) -> list[str]:
    """
    Return TWO plausible-but-false short events, max ~120 chars each.
    To avoid calling an LLM right now, use a curated pool.
    Swap this to your OpenAI call when ready.
    """
    pool = [
        "National Wednesday Ban takes effect; midweek moved to Friday.",
        "First underwater library opens to the general public.",
        "Scientists confirm the Moon briefly displayed a natural rainbow.",
        "City council standardizes tomato sizes by ordinance.",
        "Prototype metal discovered that floats on olive oil.",
        "World’s first sky-park chess tournament held on zeppelins.",
        "A town votes to replace traffic lights with choreographed dancers.",
        "New museum opens dedicated entirely to pocket lint.",
        "Meteorologists announce clouds now have official nicknames.",
        "A marathon is completed entirely on a moving train."
    ]
    random.shuffle(pool)
    # ensure the fake lines do not accidentally contain the real title fragment
    fakes = []
    for line in pool:
        if real_title.lower()[:30] not in line.lower():
            fakes.append(line)
        if len(fakes) == 2:
            break
    if len(fakes) < 2:
        fakes += ["Parliament recognizes naps as a national sport.", "World’s first reversible rainbow documented."]
    return fakes[:2]

def ensure_today_round():
    today = _now_local_date()
    if TimelineRound.query.filter_by(round_date=today).first():
        return

    real_title, real_url = pick_real_event(today)
    f1, f2 = generate_fakes_with_llm(real_title)

    row = TimelineRound(
        round_date=today,
        real_title=real_title,
        real_source_url=real_url,
        fake1_title=f1[:300],
        fake2_title=f2[:300],
        correct_index=0,  # real = index 0 pre-shuffle
    )
    db.session.add(row)
    db.session.commit()
    print(f"✅ Generated TimelineRound for {today}: {real_title}")
