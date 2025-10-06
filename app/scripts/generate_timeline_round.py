import os
import random
import datetime as dt
import requests

from app.extensions import db
from app.roulette.models import TimelineRound

# --- Wikipedia On This Day REST endpoint
WIKI_URL = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{m}/{d}"

# Tiny, safe wrapper around your LLM client; replace with your real call.
def generate_fakes_with_llm(real_title: str) -> list[str]:
    """
    Return TWO plausible-but-false short events, max ~120 chars each.
    Ensure they DO NOT repeat the real event.
    """
    # If you don't want to call out to an LLM yet, generate harmless decoys:
    samples = [
        "A royal decree officially bans umbrellas in public squares.",
        "Scientists announce the Moon briefly had a natural rainbow.",
        "World's first underwater library opens to the public.",
        "Farmers in Italy vote to standardize tomato sizes by law.",
        "A new metal is discovered that floats on olive oil."
    ]
    random.shuffle(samples)
    return samples[:2]

def pick_real_event(today: dt.date) -> tuple[str, str]:
    r = requests.get(WIKI_URL.format(m=today.month, d=today.day), timeout=20)
    r.raise_for_status()
    data = r.json()
    events = [e for e in data.get("events", []) if e.get("text") and e.get("pages")]
    # prefer short, punchy events
    events.sort(key=lambda e: len(e["text"]))
    chosen = random.choice(events[:20]) if len(events) >= 20 else random.choice(events)
    title = chosen["text"][:300]
    url = chosen["pages"][0]["content_urls"]["desktop"]["page"]
    return title, url

def ensure_today_round():
    today = dt.date.today()
    if TimelineRound.query.filter_by(round_date=today).first():
        return

    real_title, real_url = pick_real_event(today)
    fakes = generate_fakes_with_llm(real_title)
    if len(fakes) < 2:
        # fallback guards
        fakes = fakes + ["A nationwide ban on Wednesdays is announced.", "First zero-gravity marathon held in a blimp."]

    row = TimelineRound(
        round_date=today,
        real_title=real_title,
        real_source_url=real_url,
        fake1_title=fakes[0][:300],
        fake2_title=fakes[1][:300],
        correct_index=0,  # real = 0 before shuffling in UI
    )
    db.session.add(row)
    db.session.commit()
    print(f"Generated TimelineRound for {today}: {real_title}")

# If run as a script under Flask app context:
# from app import create_app
# app = create_app()
# with app.app_context():
#     ensure_today_round()
