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

def soften_real_title(title: str) -> str:
    """
    Make a real headline sound as generic as the fakes:
    - remove parentheticals
    - drop super-specific places/qualifiers
    - keep one short, neutral clause
    """
    t = (title or "").strip()

    # Remove parentheticals: "X (Y)" -> "X"
    t = re.sub(r"\s*\([^)]*\)", "", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Strip leading determiners like "The present/current"
    t = re.sub(r"^(The\s+(present|current)\s+)", "", t, flags=re.I)

    # Trim after over-specific locatives (keep first clause)
    # e.g. "… at Westminster Abbey / in London / near …"
    t = re.split(
        r"\b(?: at | in | near | within | by | during | under | between | outside | inside )\b",
        t, maxsplit=1, flags=re.I
    )[0].strip()

    # Remove appositives after comma: "X, the Y" -> "X"
    t = re.sub(r",\s*(the|a|an)\b.*$", "", t, flags=re.I)

    # Remove trailing date-y fragments like "on October 13"
    t = re.sub(r"\b(on|by|from)\s+[A-Z][a-z]+(?:\s+\d{1,2})?(?:,\s*\d{4})?$", "", t).strip()

    # Keep to one short neutral clause (cut on ";", ":")
    t = re.split(r"[;:]", t, maxsplit=1)[0].strip()

    # Gentle verb normalizations (avoid “feels older/specific” forms)
    t = re.sub(r"\bwas\b", "is", t, flags=re.I)
    t = re.sub(r"\bhave been\b", "are", t, flags=re.I)

    # Avoid screaming specifics like “present church building”
    t = re.sub(r"\b(present|current)\s+", "", t, flags=re.I)

    # Ensure it reads like an event; add light verb if needed
    if not re.search(r"\b(is|are|begins?|opens?|adopts?|declares?|formed?|founded?|launched?|consecrated|dedicated|discovered)\b", t, re.I):
        # simple fallback: append neutral verb
        t = t.rstrip(".")
        t = f"{t} occurs"

    # Final tidy
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    return t

def target_len(s: str, lo=60, hi=110) -> int:
    n = max(lo, min(hi, len(s)))
    return (lo + hi) // 2 if n < lo or n > hi else n

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
def _target_len_bounds(s: str, lo=60, hi=110):
    # keep fakes close to the real headline's length
    n = len(s.strip())
    pad = 12  # small tolerance window
    t_lo = max(50, min(hi, n - pad))
    t_hi = max(60, min(120, n + pad))
    return t_lo, t_hi

def _clean_line(x: str) -> str:
    x = x.strip().strip("•-—*").strip()
    x = re.sub(r'["“”]', "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x.rstrip(".")

def _style_like_real(fake: str, real: str) -> str:
    # Neutralize tone & punctuation; mirror capitalization style loosely
    s = _clean_line(fake)
    # avoid superlatives/dates/celebs
    s = re.sub(r"\b(first|only|greatest|famous|legendary)\b", "major", s, flags=re.I)
    s = re.sub(r"\b([A-Z][a-z]+ \d{1,2}(, \d{4})?)\b", "today", s)  # nuke specific dates
    # trim trailing qualifiers like "in London", "at X"
    s = re.split(r"\b(?: at | in | near | within | by | during | under | between )\b", s, maxsplit=1, flags=re.I)[0]
    s = s.strip().rstrip(".")
    # simple capitalization: sentence case
    if s:
        s = s[0].upper() + s[1:]
    return s

def _fallback_plausible_pool():
    # Credible, neutral, single-clause templates (no comedy)
    return [
        "A major railway line begins passenger service",
        "A new cathedral is consecrated",
        "Scientists announce a breakthrough in metal alloys",
        "An international chess exhibition is held",
        "A public museum opens to visitors",
        "A treaty is signed between European states",
        "An early experiment with flight is attempted",
        "A city introduces standardized street lighting",
        "A university marks a landmark anniversary",
        "A new bridge opens to traffic",
        "A national archive is established",
        "A civic charter is adopted by a council",
        "A scientific society is founded",
        "A postal service expands intercity routes",
        "A maritime route opens seasonal operations",
    ]

def generate_fakes_with_llm(real_title_quiz: str, n: int = 2) -> list[str]:
    """
    Produce n plausible, neutral, single-clause false headlines
    that match the style/length of the softened real title.
    Uses OpenAI if OPENAI_API_KEY is present; otherwise falls back.
    """
    tlo, thi = _target_len_bounds(real_title_quiz)
    want_note = f"{tlo}-{thi} characters, neutral tone, single clause, no names/dates."

    # --- Try OpenAI if available ---
    api_key = os.getenv("OPENAI_API_KEY")
    out: list[str] = []
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = f"""
You write quiz headlines. Create {n} plausible but false historical event one-liners.
Rules:
- Single short clause, neutral tone.
- No specific dates, no celebrity/politician names, no superlatives.
- Similar specificity and style to this real example (but different content):
  "{real_title_quiz}"
- Each line {want_note}
Return exactly {n} lines, no numbering, no quotes.
""".strip()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.6,
            )
            text = resp.choices[0].message.content or ""
            for line in text.splitlines():
                s = _style_like_real(line, real_title_quiz)
                if s and 40 <= len(s) <= 140:
                    out.append(s)
                if len(out) >= n:
                    break
        except Exception:
            out = []

    # --- Fallback or top-up from plausible pool ---
    if len(out) < n:
        pool = _fallback_plausible_pool()
        random.shuffle(pool)
        for cand in pool:
            s = _style_like_real(cand, real_title_quiz)
            if tlo <= len(s) <= thi and s.lower() != real_title_quiz.lower():
                out.append(s)
            if len(out) >= n:
                break

    # Deduplicate & pad if needed
    seen, final = set(), []
    for s in out:
        key = s.lower()
        if key not in seen and key != real_title_quiz.lower():
            seen.add(key)
            final.append(s)
        if len(final) >= n:
            break

    while len(final) < n:
        # last-resort neutral fillers near target length
        filler = _style_like_real("A regional council adopts a revised statute", real_title_quiz)
        final.append(filler)

    # add final trailing periods for consistency
    return [s.rstrip(".") + "." for s in final[:n]]

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
    print(f"[icon] Using {ai_name or name or 'fallback'} for subject={subject}")
    return _fallback_icon_name()

# ─────────────────────────────────────────────────────────────────────────────
# Main entry: ensure today's round
# ─────────────────────────────────────────────────────────────────────────────
def ensure_today_round():
    today = _now_local_date()
    existing = TimelineRound.query.filter_by(round_date=today).first()
    if existing:
        # already created for today; nothing to do
        return

    # 1) real (raw) → soften for quiz
    real_title_raw, real_url = pick_real_event(today)
    real_title_quiz = soften_real_title(real_title_raw)

    # 2) make two plausible fakes that match the style/length
    fake1_title, fake2_title = generate_fakes_with_llm(real_title_quiz, n=2)

    # 3) pick icons (use quiz-titles so keywords are in the same style)
    real_icon_name  = pick_icon_for_text(real_title_quiz)
    fake1_icon_name = pick_icon_for_text(fake1_title)
    fake2_icon_name = pick_icon_for_text(fake2_title)

    # 4) ensure icon files exist (try AI icon if missing → else fallback)
    real_icon  = _ensure_icon_exists_or_ai(real_icon_name,  real_title_quiz)
    fake1_icon = _ensure_icon_exists_or_ai(fake1_icon_name, fake1_title)
    fake2_icon = _ensure_icon_exists_or_ai(fake2_icon_name, fake2_title)

    # 5) save (correct_index=0 because the real is index 0 before shuffle in the view)
    row = TimelineRound(
        round_date=today,
        real_title=real_title_quiz,
        real_source_url=real_url,
        fake1_title=fake1_title,
        fake2_title=fake2_title,
        real_icon=real_icon,
        fake1_icon=fake1_icon,
        fake2_icon=fake2_icon,
        correct_index=0,
    )
    db.session.add(row)
    db.session.commit()
    print(f"✅ Generated TimelineRound for {today}: {real_title_quiz}")
