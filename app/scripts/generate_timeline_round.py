# app/scripts/generate_timeline_round.py
import os, re, random, time, datetime as dt, requests, pytz
from urllib.parse import quote_plus

from app.extensions import db
from app.roulette.models import TimelineRound

TZ = os.getenv("TR_TZ", "America/Denver")
USER_AGENT = os.getenv("TR_UA", "TimelineRoulette/1.0 (+https://therealroundup.com)")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")  # set this in Render

WIKI_REST = "https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{m}/{d}"
UNSPLASH_SEARCH = "https://api.unsplash.com/search/photos?query={q}&orientation=squarish&per_page=1"

STOPWORDS = set(("the a an of and for to in on at with from into during across over under new old first second third"
                 " king queen war treaty republic empire national world city state country museum library university"
                 " begins founded formed opens launch launches announces announces").split())

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

def pick_real_event(today: dt.date) -> tuple[str, str]:
    data = _http_get_json(WIKI_REST.format(m=today.month, d=today.day))
    events = [e for e in data.get("events", []) if e.get("text") and e.get("pages")]
    if not events:
        return ("On this day: a notable historical event occurred.", "https://en.wikipedia.org/wiki/Main_Page")
    events.sort(key=lambda e: len(e.get("text", "")))
    chosen = random.choice(events[:20]) if len(events) >= 20 else random.choice(events)
    title = chosen["text"][:300]
    url = chosen["pages"][0]["content_urls"]["desktop"]["page"]
    return title, url

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

def extract_keyword(text: str) -> str:
    # Very light keyword picker: keep nouns-ish by dropping stopwords and short tokens
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    for w in words:
        if w not in STOPWORDS:
            return w
    return words[0] if words else "history"

def unsplash_thumb(query: str) -> tuple[str | None, str | None]:
    """
    Returns (image_url, attribution_text) or (None, None) if not available.
    Attribution text example: "John Smith • https://unsplash.com/@johnsmith"
    """
    if not UNSPLASH_ACCESS_KEY:
        return None, None
    url = UNSPLASH_SEARCH.format(q=quote_plus(query))
    data = _http_get_json(url, headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"})
    results = data.get("results", [])
    if not results:
        return None, None
    r0 = results[0]
    # prefer a small square/cropped URL if available
    img = r0.get("urls", {}).get("small_s3") or r0.get("urls", {}).get("small") or r0.get("urls", {}).get("thumb")
    user = r0.get("user", {})
    name = user.get("name") or user.get("username") or "Unsplash"
    profile = user.get("links", {}).get("html") or "https://unsplash.com"
    return img, f"{name} • {profile}"

def ensure_today_round():
    today = _now_local_date()
    if TimelineRound.query.filter_by(round_date=today).first():
        return

    real_title, real_url = pick_real_event(today)
    f1, f2 = generate_fakes_with_llm(real_title)

    # fetch thumbnails (free)
    real_kw = extract_keyword(real_title)
    f1_kw = extract_keyword(f1)
    f2_kw = extract_keyword(f2)

    real_img, real_attr = unsplash_thumb(real_kw)
    f1_img, f1_attr = unsplash_thumb(f1_kw)
    f2_img, f2_attr = unsplash_thumb(f2_kw)

    row = TimelineRound(
        round_date=today,
        real_title=real_title, real_source_url=real_url,
        fake1_title=f1[:300], fake2_title=f2[:300],
        real_img_url=real_img, real_img_attr=real_attr,
        fake1_img_url=f1_img, fake1_img_attr=f1_attr,
        fake2_img_url=f2_img, fake2_img_attr=f2_attr,
        correct_index=0,
    )
    db.session.add(row); db.session.commit()
    print(f"✅ Generated TimelineRound for {today}: {real_title}")
