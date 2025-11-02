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


# --------- similarity + token helpers ----------
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

def _words(s: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(s or "")]

def _token_set(s: str) -> set[str]:
    # only meaningful tokens (>=4 chars) to avoid overlap on "in/with/of"
    return {w for w in _words(s) if len(w) >= 4}

def _trigrams(s: str) -> set[tuple[str, str, str]]:
    ws = _words(s)
    return {(ws[i], ws[i+1], ws[i+2]) for i in range(max(0, len(ws)-2))}

def _too_similar(a: str, b: str) -> bool:
    """
    Return True if a and b are 'too similar':
    - token Jaccard > 0.55 OR
    - trigram Jaccard > 0.40
    """
    if not a or not b:
        return False
    ta, tb = _token_set(a), _token_set(b)
    if ta and tb:
        j = len(ta & tb) / max(1, len(ta | tb))
        if j > 0.55:
            return True
    ga, gb = _trigrams(a), _trigrams(b)
    if ga and gb:
        jg = len(ga & gb) / max(1, len(ga | gb))
        if jg > 0.40:
            return True
    return False

def _wlen(s: str) -> int:
    return len(WORD_RE.findall(s or ""))

def _clean_terminal_punct(s: str) -> str:
    s = (s or "").strip()
    s = s.rstrip("…").rstrip(" .,:;—–-") + "."
    s = re.sub(r"\s{2,}", " ", s)
    return s

def _equalize_length(text: str, target_words: int, rng: Optional[random.Random] = None) -> str:
    """
    Bring `text` to within ±3 words of target length without obvious repetition.
    """
    rng = rng or random.Random()
    t = _clean_terminal_punct(text)

    MAX_OVER = 3
    MIN_UNDER = 3

    # If too long: trim by clauses, then words
    while _wlen(t) > target_words + MAX_OVER:
        # drop trailing clause after common joiners
        cut = re.split(r"(?:,?\s(?:after|as|to|for|while|following|amid|because|which)\b.*)$", t, 1, flags=re.I)[0]
        if cut and _wlen(cut) >= target_words - MAX_OVER:
            t = _clean_terminal_punct(cut); break
        cut = re.split(r"[—–-]|,(?![^()]*\))", t)[0]
        if cut and _wlen(cut) >= target_words - MAX_OVER:
            t = _clean_terminal_punct(cut); break
        t = _clean_terminal_punct(" ".join(t.split()[:-1] or t.split()))

    # If too short: add a short, neutral **one-time** tail
    if _wlen(t) < target_words - MIN_UNDER:
        tails = [
            "after tests.", "for a wider rollout.", "to boost creators.",
            "for a global audience.", "after a pilot run.", "to align with new rules.",
            "following a close vote.", "with a software update.", "in select regions."
        ]
        rng.shuffle(tails)
        for tail in tails:
            if tail.lower() not in t.lower():  # prevent duplicates like "today to align…"
                cand = _clean_terminal_punct(t[:-1] + " " + tail)
                if abs(_wlen(cand) - target_words) <= 3:
                    t = cand
                    break

    # final tidy and guardrails
    t = _clean_terminal_punct(t)
    if _wlen(t) > target_words + MAX_OVER:
        t = _clean_terminal_punct(" ".join(t.split()[:target_words + MAX_OVER]))
    elif _wlen(t) < target_words - MIN_UNDER:
        t = _clean_terminal_punct(t + " worldwide.")
    return t

# ---------------- Youth relevance helpers ----------------
POP_KEYWORDS = {
    "social_media": ["tiktok","instagram","facebook","twitter","x.com","snapchat","youtube","reddit","twitch","vine","myspace"],
    "gaming": ["nintendo","playstation","xbox","pokemon","minecraft","fortnite","roblox","steam","esports","blizzard","sony"],
    "music": ["spotify","itunes","mtv","grammy","billboard","taylor swift","drake","bts","k-pop","eminem","nirvana"],
    "film_tv": ["netflix","disney","marvel","star wars","hbo","pixar","oscars","imdb","hulu","prime video","anime"],
    "tech": ["iphone","android","apple","google","microsoft","ai","openai","tesla","spacex","nasa","satellite","internet","www"],
    "sports": ["nba","nfl","mlb","nhl","fifa","world cup","olympics","super bowl","ucla","alabama","yankees","lakers"],
    "internet_culture": ["meme","viral","hashtag","emoji","stream","podcast","blog","wiki","open source","linux","browser"],
}

def _infer_domain(text: str) -> str:
    lc = (text or "").lower()
    for domain, words in POP_KEYWORDS.items():
        if any(w in lc for w in words):
            return domain
    # backoffs
    if any(k in lc for k in ("app","website","online","platform","streaming")): return "internet_culture"
    if any(k in lc for k in ("rocket","mission","launch","orbit")): return "space"
    return "general"

def _score_for_youth(event_obj: dict) -> float:
    """
    Score a Wikipedia OTD event for 'youth appeal':
    + keywords in pop domains
    + recency (post-1980 weighted up)
    + concise (<= 140 chars ideal)
    """
    t = (event_obj.get("text") or event_obj.get("displaytitle") or "").strip()
    if not t:
        return -1e9
    lc = t.lower()

    # keyword score
    kw_score = 0.0
    for words in POP_KEYWORDS.values():
        if any(w in lc for w in words):
            kw_score += 2.0

    # recency score (Wikipedia OTD has 'year')
    year = event_obj.get("year")  # often present
    recency = 0.0
    try:
        y = int(year)
        if y >= 2010:   recency = 3.0
        elif y >= 2000: recency = 2.0
        elif y >= 1980: recency = 1.0
    except Exception:
        recency = 0.0

    # concision / readability
    L = len(t)
    concision = 1.0 if 60 <= L <= 140 else (0.3 if L <= 180 else -0.5)

    # penalize dry bureaucratic verbs
    if re.search(r"\b(henceforth|thereof|therein|whereby)\b", lc):
        concision -= 0.5

    return kw_score + recency + concision

def _youthify_title(text: str) -> str:
    """
    Make copy punchier for cards: <~120 chars, plain words, active voice.
    """
    if not text:
        return ""
    t = _soften_real_title(text)

    # remove parenthetical clutter
    t = re.sub(r"\s*\([^)]{0,80}\)", "", t)

    # friendlier verbs
    replacements = {
        r"\bannounces\b": "reveals",
        r"\bauthorizes\b": "OKs",
        r"\bcommissions\b": "greenlights",
        r"\bcommences\b": "starts",
        r"\bintroduces\b": "launches",
        r"\butilizes\b": "uses",
        r"\bpermits\b": "lets",
    }
    for pat, sub in replacements.items():
        t = re.sub(pat, sub, t, flags=re.I)

    # trim to ~140 but avoid ellipsis (we need stable word counts)
    limit = 140
    if len(t) > limit:
        cut = t[:limit].rsplit(" ", 1)[0].rstrip(",.;:- ")
        if cut:
            t = cut  # no "…"

    # micro-style: drop leading year dash if present ("2009 – Facebook…")
    t = re.sub(r"^\d{3,4}\s*[—–-]\s*", "", t)

    return t.strip()

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

def _openai_fakes_from_real(real_text: str, month_name: str, target_words: Optional[int] = None) -> Tuple[str, str]:
    """
    Two plausible-but-false entries:
    - same domain vibe as real
    - within ±3 words of target
    - NOT too similar to the real or to each other
    """
    domain = _infer_domain(real_text)
    target_words = target_words or _wlen(real_text)
    rng = random.Random(int(datetime.now(TZ).strftime("%Y%m%d")) ^ (hash(real_text) & 0xFFFFFFFF))

    # ---------- 1) Try OpenAI with constraints ----------
    if OPENAI_API_KEY:
        sys_prompt = (
            "Write short, believable 'On This Day' entries for social media.\n"
            f"Month: {month_name}. Domain: {domain} (e.g., social media, gaming, music, film/TV, sports, tech).\n"
            f"Each entry MUST be {target_words} words ±3, include a proper noun + a 4-digit year, "
            "and use a different subject/entity than the real entry.\n"
            "Avoid repeating phrases, avoid hedging/meta. Output STRICT JSON: {\"fake1\":\"...\",\"fake2\":\"...\"}"
        )
        payload = {
            "model": OAI_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Real entry: {real_text}"},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        try:
            r = requests.post(OAI_URL,
                              headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                              json=payload, timeout=30)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            try:
                obj = requests.utils.json.loads(content)
            except Exception:
                obj = requests.utils.json.loads(content.strip("` \n"))
            cands = [obj.get("fake1","").strip(), obj.get("fake2","").strip()]
            normed: list[str] = []
            for s in cands:
                s = _clean_terminal_punct(s)
                s = _equalize_length(s, target_words, rng)
                if s and not _too_similar(s, real_text):
                    normed.append(s)
            if len(normed) >= 2 and not _too_similar(normed[0], normed[1]):
                return normed[0], normed[1]
        except Exception:
            time.sleep(0.4)

    # ---------- 2) Deterministic fallback with diversity ----------
    banks = {
        "social_media": (["Instagram","TikTok","YouTube","Twitter","Snapchat","Reddit"],
                         ["rolls out","debuts","tests","expands","rebrands","enables"],
                         ["creator fund","short-form tool","live shopping","DM encryption","music library","moderation update"],
                         ["Los Angeles","Seoul","London","New York","Tokyo","Berlin"]),
        "gaming": (["Nintendo","Sony","Microsoft","Blizzard","Valve","Epic Games"],
                   ["announces","releases","patches","launches","revives","updates"],
                   ["handheld console","cross-play feature","online service","battle pass","esports circuit","mod support"],
                   ["Kyoto","Seattle","Helsinki","Montreal","Osaka","Austin"]),
        "music": (["Spotify","MTV","Billboard","Grammy Academy","Apple Music","SoundCloud"],
                  ["introduces","launches","rebrands","expands","revamps","opens"],
                  ["global chart","streaming tier","award category","curator program","student plan","royalty model"],
                  ["Stockholm","Los Angeles","Nashville","London","Seoul","Berlin"]),
        "film_tv": (["Netflix","Disney","HBO","Prime Video","Hulu","Crunchyroll"],
                    ["premieres","unveils","greenlights","adds","bundles","licenses"],
                    ["original series","ad-supported plan","download option","anime slate","sports docuseries","student bundle"],
                    ["Burbank","Tokyo","Toronto","Madrid","Mumbai","Sydney"]),
        "tech": (["Apple","Google","Microsoft","Samsung","NVIDIA","OpenAI"],
                 ["releases","announces","ships","open-sources","unveils","rolls out"],
                 ["smartphone update","AI toolkit","browser feature","cloud plan","GPU program","assistant upgrade"],
                 ["Cupertino","Seoul","Mountain View","Redmond","Taipei","Shenzhen"]),
        "sports": (["FIFA","NBA","NFL","IOC","UEFA","MLB"],
                   ["confirms","awards","announces","expands","adopts","approves"],
                   ["host city","play-in format","salary cap rules","streaming deal","ref review system","wild-card slot"],
                   ["Zurich","Paris","New York","Dallas","Doha","Tokyo"]),
        "internet_culture": (["Reddit","Twitch","Discord","Wikipedia","GitHub","WordPress"],
                             ["adds","pilots","disables","restores","launches","overhauls"],
                             ["awards program","streaming tool","mod tools","dark mode","sponsorships","plugin store"],
                             ["San Francisco","Vancouver","Dublin","Singapore","Austin","Amsterdam"]),
        "general": (["NASA","SpaceX","UNESCO","BBC","ESA","CERN"],
                    ["announces","opens","tests","adopts","extends","funds"],
                    ["mission program","broadcast rule","education grant","archive","lab upgrade","satellite service"],
                    ["Houston","Cape Canaveral","Paris","Geneva","Tokyo","Bangalore"]),
    }
    ORGS, VERBS, OBJS, PLACES = banks.get(domain, banks["general"])
    real_tokens = _token_set(real_text)

    def compose() -> str:
        # ensure different subject than the real one
        choices = [o for o in ORGS if o.lower() not in real_tokens] or ORGS
        org, verb, obj, place = rng.choice(choices), rng.choice(VERBS), rng.choice(OBJS), rng.choice(PLACES)
        year = rng.randint(1980, 2022)
        s = f"{org} {verb} {obj} in {year} at {place}."
        s = _equalize_length(_clean_terminal_punct(s), target_words, rng)
        return s

    # rejection sampling for diversity
    fakes: list[str] = []
    attempts = 0
    while len(fakes) < 2 and attempts < 20:
        attempts += 1
        cand = compose()
        if _too_similar(cand, real_text):
            continue
        if any(_too_similar(cand, x) for x in fakes):
            continue
        if cand not in fakes:
            fakes.append(cand)

    if len(fakes) < 2:
        # guaranteed, distinct-ish backup
        while len(fakes) < 2:
            fakes.append(_equalize_length(compose() + " Update follows testing.", target_words, rng))

    return fakes[0], fakes[1]

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

    # 1) score for youth appeal and keep top few
    scored = sorted(
        [e for e in events if (e.get("text") or e.get("displaytitle"))],
        key=_score_for_youth,
        reverse=True,
    )

    # 2) choose the best concise candidate; fallback to generic heuristics
    picked_text = None
    for e in scored[:12]:
        t = (e.get("text") or e.get("displaytitle") or "").strip()
        if 40 <= len(t) <= 180 and t.count(".") <= 1:
            picked_text = t
            break

    if not picked_text:
        # original fallback
        random.shuffle(events)
        for e in events:
            t = (e.get("text") or e.get("displaytitle") or "").strip()
            if 40 <= len(t) <= 180 and t.count(".") <= 1:
                picked_text = t
                break

    if not picked_text:
        picked_text = "A notable pop-culture or tech milestone is recorded."

    # month name for prompting
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
    real_soft = _youthify_title(real_raw)   # or _soften_real_title()
    target_words = _wlen(real_soft)

    fake1, fake2 = _openai_fakes_from_real(real_soft, month_name, target_words)
    fake1 = _equalize_length(_youthify_title(fake1), target_words)
    fake2 = _equalize_length(_youthify_title(fake2), target_words)

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