# app/scripts/icon_ai.py
import os, re
from pathlib import Path
from datetime import datetime

# --- OpenAI client (Python SDK) ---
# Make sure OPENAI_API_KEY is set in your environment (Render → Environment)
from openai import OpenAI
client = OpenAI()

ICON_DIR = Path(__file__).resolve().parents[1] / "static" / "roulette" / "icons"

# very small, safe SVG policy
ALLOWED_TAGS = {"svg","path","circle","rect","line","polyline","polygon","g"}
ALLOWED_ATTR = {
    "svg": {"xmlns","width","height","viewBox","fill","stroke","stroke-width","stroke-linecap","stroke-linejoin"},
    "path": {"d","fill","stroke","stroke-width","stroke-linecap","stroke-linejoin"},
    "circle": {"cx","cy","r","fill","stroke","stroke-width","stroke-linecap","stroke-linejoin"},
    "rect": {"x","y","width","height","rx","ry","fill","stroke","stroke-width"},
    "line": {"x1","y1","x2","y2","stroke","stroke-width","stroke-linecap"},
    "polyline": {"points","fill","stroke","stroke-width","stroke-linecap","stroke-linejoin"},
    "polygon": {"points","fill","stroke","stroke-width","stroke-linecap","stroke-linejoin"},
    "g": {"fill","stroke","stroke-width","stroke-linecap","stroke-linejoin"},
}

def _sanitize_svg(svg: str) -> str | None:
    # quick checks
    if "<svg" not in svg or "</svg>" not in svg:
        return None
    # forbid scripts, images, external refs, styles
    forbidden = ("<script", "<image", "xlink:", "href=", "<style", "<?xml", "<!DOCTYPE", "<foreignObject")
    if any(tok.lower() in svg.lower() for tok in forbidden):
        return None

    # normalize basic outer svg
    # ensure stroke="currentColor" and no fixed colors
    svg = re.sub(r'stroke="#[0-9A-Fa-f]{3,6}"', 'stroke="currentColor"', svg)
    svg = re.sub(r'fill="#[0-9A-Fa-f]{3,6}"', 'fill="none"', svg)

    # force attributes (viewBox & size)
    svg = re.sub(r'<svg\b[^>]*>', '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">', svg)

    # super-lightweight structural sanity (don’t allow tags we didn’t list)
    tags = re.findall(r'</?([a-zA-Z0-9]+)\b', svg)
    for t in tags:
        t = t.lower()
        if t not in ALLOWED_TAGS:
            return None

    return svg

_PROMPT = """You are an icon generator. Output ONLY a single compact SVG element (no prose).
Requirements:
- Minimal outline icon, 24x24 viewBox, clean single-color lines (stroke="currentColor", fill="none").
- No external references, no <style>, no scripts, no text, no raster images.
- Prefer a single <path> plus simple shapes. Keep it ~200-800 characters.
Subject: a simple, easily recognizable icon representing: "{subject}"
Return ONLY the <svg>...</svg> markup.
"""

def generate_icon_svg(subject: str) -> str | None:
    # Ask a small, fast model; adjust to your default (e.g., "gpt-4o-mini" or similar)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You produce minimal, safe SVG icons."},
            {"role": "user", "content": _PROMPT.format(subject=subject)},
        ]
    )
    svg = completion.choices[0].message.content.strip()
    return _sanitize_svg(svg)

def ensure_ai_icon(filename: str, subject: str) -> str | None:
    """
    Create icons/ai-<filename>.svg if it doesn't exist, using the subject phrase, and return name.
    Returns the basename (e.g., "ai-train.svg") or None on failure.
    """
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"ai-{filename}"
    out_path = ICON_DIR / basename
    if out_path.exists():
        return basename

    svg = generate_icon_svg(subject)
    if not svg:
        return None
    out_path.write_text(svg, encoding="utf-8")
    return basename
