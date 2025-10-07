# app/scripts/generate_timeline_round.py
import os, re, random, time
import datetime as dt
from pathlib import Path

import pytz
import requests

from app.extensions import db
from app.roulette.models import TimelineRound

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TZ = os.getenv("TR_TZ", "America/Denver")
USER_AGENT = os.getenv("TR_UA", "TimelineRoulette/1.0 (+https://therealroundup.com)")
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{m}/{d}"

# where your icons live
ICON_DIR = Path(__file__).resolve().parents[1] / "static" / "roulette" / "icons"
ICON_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _now_local_date() -> dt.date:
    return dt.datetime.now(pytz.timezone(TZ)).date()

def _http_get_json(url: str, headers: dict | None = None, max_retries: int = 3, timeout: int = 20) -> dict:
    base_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        base_headers.update(headers)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=base_headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(1.0 * attempt, 3.0))
    raise last_err

# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia → real event
# ─────────────────────────────────────────────────────────────────────────────
def pick_real_event(today: dt.date) -> tuple[str, str]:
    data = _http_get_json(WIKI_REST.format(m=today.month, d=today.day))
    events = [e for e in data.get("events", []) if e.get("text") and e.get("pages")]
    if not events:
        return ("On this day: a notable historical event occurred.", "https://en.wikipedia.org/wiki/Main_Page")
    # prefer shorter, punchy entries
    events.sort(key=lambda e: len(e.get("text","")))
    chosen = random.choice(events[:20]) if len(events) >= 20 else events[0]
    title = chosen["text"][:300]
    url = chosen["pages"][0]["content_urls"]["desktop"]["page"]
    return title, url

# ─────────────────────────────────────────────────────────────────────────────
# Fake events (tiny, plausible-but-false)
# ─────────────────────────────────────────────────────────────────────────────
def generate_fakes_with_llm(real_title: str) -> list[str]:
    pool = [
        "National Wednesday Ban takes effect; midweek moved to Friday.",
        "First underwater library opens to the general public.",
        "Scientists confirm the Moon briefly displayed a natural rainbow.",
        "City council standardizes tomato sizes by ordinance.",
        "Prototype metal discovered that floats on olive oil.",
        "World’s first sky-park chess tournament held on zeppelins.",
        "Town replaces traffic lights with choreographed dancers.",
        "New museum opens dedicated entirely to pocket lint.",
        "Meteorologists announce clouds now have official nicknames.",
        "A marathon is completed entirely on a moving train."
    ]
    random.shuffle(pool)
    fakes = []
    for line in pool:
        if real_title.lower()[:30] not in line.lower():
            fakes.append(line)
        if len(fakes) == 2:
            break
    if len(fakes) < 2:
        fakes += ["Parliament recognizes naps as a national sport.", "World’s first reversible rainbow documented."]
    return fakes[:2]

# ─────────────────────────────────────────────────────────────────────────────
# Smart icon chooser that adapts to whatever SVGs you have
# (from your message; lightly integrated)
# ─────────────────────────────────────────────────────────────────────────────
# common neutral icons we can safely use as fallback (first one that exists wins)
FALLBACK_CANDIDATES = [
    "star.svg", "sparkles.svg", "asterisk.svg", "dot.svg", "circle.svg", "history.svg",
    "compass.svg", "feather.svg"
]

def _fallback_icon_name() -> str:
    for name in FALLBACK_CANDIDATES:
        if (ICON_DIR / name).exists():
            return name
    # last resort: pick ANY svg in the folder
    for p in ICON_DIR.glob("*.svg"):
        return p.name
    return "star.svg"  # if folder is empty (shouldn't happen)

FALLBACK_ICON = _fallback_icon_name()

# multiple plausible filenames for each concept (we pick the first that exists)
KEYWORD_ICON_CANDIDATES: dict[str, list[str]] = {
    # transport
    "train": ["train.svg", "tram.svg", "subway.svg", "locomotive.svg"],
    "rail":  ["train.svg", "tram.svg", "railway.svg"],
    "railway": ["train.svg", "tram.svg"],
    "ship": ["ship.svg", "boat.svg", "ferry.svg"],
    "navy": ["ship.svg", "anchor.svg"],
    "sail": ["ship.svg", "sailboat.svg"],
    "plane": ["plane.svg", "airplane.svg", "jet.svg"],
    "aviation": ["plane.svg", "airplane.svg"],
    "flight": ["plane.svg", "airplane.svg"],
    "zeppelin": ["airship.svg", "blimp.svg", "balloon.svg", "plane.svg"],
    "bridge": ["bridge.svg", "landmark.svg", "building.svg"],
    "canal": ["bridge.svg", "water.svg"],
    "tower": ["landmark.svg", "building.svg"],
    "cathedral": ["landmark.svg", "building.svg", "church.svg"],

    # politics / conflict
    "war": ["swords.svg", "sword.svg", "shield.svg", "flag.svg"],
    "battle": ["swords.svg", "shield.svg", "flag.svg"],
    "revolt": ["swords.svg", "flag.svg"],
    "treaty": ["scroll.svg", "file-text.svg", "file.svg"],
    "republic": ["flag.svg", "government.svg"],
    "election": ["ballot.svg", "vote.svg", "check-square.svg"],
    "constitution": ["scroll.svg", "book.svg", "file-text.svg"],
    "parliament": ["ballot.svg", "flag.svg", "building.svg"],
    "king": ["crown.svg", "flag.svg"],
    "queen": ["crown.svg", "flag.svg"],
    "empire": ["flag.svg", "globe.svg"],

    # science / tech
    "scientist": ["flask.svg", "beaker.svg", "test-tube.svg", "lab.svg"],
    "laboratory": ["flask.svg", "beaker.svg", "test-tube.svg", "lab.svg"],
    "experiment": ["flask.svg", "beaker.svg", "lightbulb.svg"],
    "physics": ["flask.svg", "atom.svg", "lightbulb.svg"],
    "chemistry": ["flask.svg", "beaker.svg", "test-tube.svg"],
    "computer": ["cpu.svg", "monitor.svg", "laptop.svg"],
    "internet": ["globe.svg", "network.svg"],
    "satellite": ["satellite.svg", "antenna.svg"],
    "telegraph": ["cpu.svg", "radio.svg", "antenna.svg"],
    "metal": ["cube.svg", "box.svg"],
    "alloy": ["cube.svg", "box.svg"],
    "discovery": ["lightbulb.svg", "search.svg", "compass.svg"],
    "invention": ["lightbulb.svg", "wrench.svg"],

    # culture / sports
    "music": ["music.svg", "music-2.svg", "headphones.svg"],
    "symphony": ["music.svg"],
    "band": ["music.svg", "music-2.svg"],
    "museum": ["landmark.svg", "building.svg"],
    "library": ["book.svg", "library.svg"],
    "book": ["book.svg"],
    "chess": ["chess.svg", "chess-knight.svg"],
    "game": ["joystick.svg", "gamepad.svg", "dice.svg"],
    "olympic": ["medal.svg", "trophy.svg"],
    "tournament": ["medal.svg", "trophy.svg"],
}

# broad categories (used as a second pass)
CATEGORY_CANDIDATES = {
    "transport": ["compass.svg", "map.svg", "navigation.svg", "road.svg"],
    "politics":  ["flag.svg", "building.svg", "scales.svg"],
    "science":   ["lightbulb.svg", "atom.svg", "flask.svg"],
    "culture":   ["star.svg", "music.svg", "trophy.svg"],
}

TRANSPORT = {"train","rail","railway","tram","ship","navy","sail","plane","aviation","flight","airship","zeppelin","bridge","canal"}
POLITICS  = {"war","battle","treaty","republic","election","constitution","parliament","king","queen","empire"}
SCIENCE   = {"scientist","laboratory","experiment","physics","chemistry","computer","internet","satellite","telegraph","metal","alloy","discovery","invention"}
CULTURE   = {"music","symphony","band","museum","library","book","chess","game","olympic","tournament"}

STOP = {
    "the","a","an","of","and","for","to","in","on","at","with","from","into","during","across","over","under",
    "new","old","first","second","third","year","day","city","state","country","national","world","begins","is","are",
    "was","were","becomes","formed","opens","announces"
}

def _first_existing(candidates: list[str]) -> str | None:
    for name in candidates:
        if (ICON_DIR / name).exists():
            return name
    return None

def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower()) if w not in STOP]

def pick_icon_for_text(text: str) -> str:
    # 1) exact keyword → candidates
    for w in _words(text):
        cand = KEYWORD_ICON_CANDIDATES.get(w)
        if cand:
            name = _first_existing(cand)
            if name: return name

    ws = set(_words(text))
    # 2) category fallbacks
    if ws & TRANSPORT:
        name = _first_existing(CATEGORY_CANDIDATES["transport"])
        if name: return name
    if ws & POLITICS:
        name = _first_existing(CATEGORY_CANDIDATES["politics"])
        if name: return name
    if ws & SCIENCE:
        name = _first_existing(CATEGORY_CANDIDATES["science"])
        if name: return name
    if ws & CULTURE:
        name = _first_existing(CATEGORY_CANDIDATES["culture"])
        if name: return name

    # 3) universal fallback that we know exists
    return _fallback_icon_name()

# ─────────────────────────────────────────────────────────────────────────────
# Optional: AI SVG creation (uses free OpenAI tokens) if mapped icon is missing
# ─────────────────────────────────────────────────────────────────────────────
def _sanitize_svg(svg: str) -> str | None:
    # Basic guardrails
    if "<svg" not in svg or "</svg>" not in svg:
        return None
    forbidden = ("<script", "<image", "xlink:", "href=", "<style", "<?xml", "<!DOCTYPE", "<foreignObject")
    ls = svg.lower()
    if any(tok in ls for tok in forbidden):
        return None

    # Normalize minimal attributes & colors
    svg = re.sub(r'stroke="#[0-9A-Fa-f]{3,6}"', 'stroke="currentColor"', svg)
    svg = re.sub(r'fill="#[0-9A-Fa-f]{3,6}"', 'fill="none"', svg)
    svg = re.sub(
        r'<svg\b[^>]*>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">',
        svg,
        count=1
    )

    # Lightweight tag allowlist
    tags = re.findall(r'</?([a-zA-Z0-9]+)\b', svg)
    allowed = {"svg","path","circle","rect","line","polyline","polygon","g"}
    for t in tags:
        if t.lower() not in allowed:
            return None
    return svg

def _openai_client_or_none():
    try:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAI()
    except Exception:
        return None

_ICON_PROMPT = """You are an icon generator. Output ONLY a single compact SVG element (no prose).
Requirements:
- Minimal outline icon, 24x24 viewBox, clean single-color lines (stroke="currentColor", fill="none").
- No external references, no <style>, no scripts, no text, no raster images.
- Prefer a single <path> plus simple shapes. Keep it ~200-800 characters.
Subject: a simple, easily recognizable icon representing: "{subject}"
Return ONLY the <svg>...</svg> markup.
"""

def _ensure_ai_icon(filename: str, subject: str) -> str | None:
    """
    If icons/ai-<filename> doesn't exist, ask OpenAI to create one and save it.
    Returns the basename (e.g., "ai-train.svg") or None on failure.
    """
    ai_basename = f"ai-{filename}"
    out_path = ICON_DIR / ai_basename
    if out_path.exists():
        return ai_basename

    client = _openai_client_or_none()
    if not client:
        return None

    try:
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_ICON_MODEL", "gpt-4o-mini"),  # small & cheap
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You produce minimal, safe SVG icons."},
                {"role": "user", "content": _ICON_PROMPT.format(subject=subject)},
            ],
        )
        svg = (completion.choices[0].message.content or "").strip()
        svg = _sanitize_svg(svg)
        if not svg:
            return None
        out_path.write_text(svg, encoding="utf-8")
        return ai_basename
    except Exception:
        return None

def _ensure_icon_exists_or_ai(name: str | None, subject: str) -> str:
    """
    If 'name' exists in ICON_DIR, return it.
    Otherwise try to create ai-<name> with OpenAI. If that fails, return a fallback that exists.
    """
    if name and (ICON_DIR / name).exists():
        return name
    base = (name or "generic.svg").replace(".svg", "") + ".svg"
    ai_name = _ensure_ai_icon(base, subject)
    if ai_name and (ICON_DIR / ai_name).exists():
        return ai_name
    # last resort
    return _fallback_icon_name()

# ─────────────────────────────────────────────────────────────────────────────
# Main entry: ensure today's round
# ─────────────────────────────────────────────────────────────────────────────
def ensure_today_round():
    today = _now_local_date()
    if TimelineRound.query.filter_by(round_date=today).first():
        return

    real_title, real_url = pick_real_event(today)
    f1, f2 = generate_fakes_with_llm(real_title)

    # choose icons, create AI icons only if needed
    real_icon  = _ensure_icon_exists_or_ai(pick_icon_for_text(real_title), real_title)
    fake1_icon = _ensure_icon_exists_or_ai(pick_icon_for_text(f1),         f1)
    fake2_icon = _ensure_icon_exists_or_ai(pick_icon_for_text(f2),         f2)

    row = TimelineRound(
        round_date=today,
        real_title=real_title,
        real_source_url=real_url,
        fake1_title=f1[:300],
        fake2_title=f2[:300],
        real_icon=real_icon,
        fake1_icon=fake1_icon,
        fake2_icon=fake2_icon,
        correct_index=0,  # real = index 0 before shuffle on the view
    )
    db.session.add(row)
    db.session.commit()
    print(f"✅ Generated TimelineRound for {today}: {real_title}")
