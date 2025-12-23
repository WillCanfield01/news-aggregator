from __future__ import annotations

import re
from typing import Tuple


def _extract_photo_credit(text: str) -> tuple[str | None, str]:
    """Extract the first photo credit line and return (credit, remaining_text)."""
    credit = None
    body = text or ""
    m = re.search(r"photo by[^\\n]*", body, flags=re.I)
    if m:
        credit = m.group(0).strip()
        body = body[: m.start()] + body[m.end() :]
    return credit, body


def _strip_markdown_noise(text: str) -> str:
    t = text or ""
    lines = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\s*[-*+]\s+", line):
            continue  # drop bullets entirely
        if re.match(r"^#{1,6}\s+", line):
            continue  # drop headings
        if re.match(r"(?i)photo by", line):
            continue
        if re.match(r"(?i)what to do:?", line):
            continue
        lines.append(line)
    t = " ".join(lines)
    t = re.sub(r"\*", " ", t)
    t = re.sub(r"[|_`~]", " ", t)
    t = re.sub(r"[—–]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _keep_first_what_happened(text: str) -> str:
    """
    If "What happened" exists, keep that paragraph and drop the rest;
    otherwise return original text.
    """
    lower = text.lower()
    idx = lower.find("what happened")
    if idx == -1:
        return text
    snippet = text[idx:]
    # stop at next "what to do" or next heading-ish cue
    stop_tokens = ["what to do", "why it matters", "key takeaways", "remediation"]
    stop_idx = len(snippet)
    low_snip = snippet.lower()
    for tok in stop_tokens:
        pos = low_snip.find(tok)
        if pos != -1 and pos < stop_idx:
            stop_idx = pos
    return snippet[:stop_idx].strip()


def _sentences_until_limit(text: str, limit: int = 520) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        chunk = text[:limit].rstrip(" .,!?:;")
        return chunk + "."
    out = []
    total = 0
    for s in sentences:
        if total + len(s) > limit:
            break
        out.append(s)
        total += len(s) + 1
        if len(out) >= 3:
            break
    combined = " ".join(out).strip()
    if not combined.endswith((".", "!", "?")):
        last = max(combined.rfind("."), combined.rfind("!"), combined.rfind("?"))
        if last != -1:
            combined = combined[: last + 1]
        elif len(combined) > 240:
            combined = combined[:240].rstrip(" .,!?:;") + "."
    return combined


def build_preview_from_text(full_text: str) -> Tuple[str | None, str]:
    """
    Create a clean free-preview summary:
    - Remove photo credit and return it separately
    - Drop bullets/headings and markdown noise
    - Keep only the first 'What happened' paragraph if present; drop 'What to do'
    - 1-2 short paragraphs, 420-520 chars max, ending on a sentence
    """
    credit, body = _extract_photo_credit(full_text or "")
    cleaned = _strip_markdown_noise(body)
    cleaned = _keep_first_what_happened(cleaned)
    trimmed = _sentences_until_limit(cleaned, limit=520)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", trimmed) if p.strip()] or [trimmed]
    preview = "\n\n".join(paragraphs[:2])
    return credit, preview
