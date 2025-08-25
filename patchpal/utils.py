# patchpal/utils.py
from __future__ import annotations
import re

def _fmt_badges(item):
    parts = []

    # Severity / CVSS -> badge
    sev = (item.get("severity") or "").upper()
    cvss = item.get("cvss") or item.get("cvss_score")
    if sev in ("CRITICAL", "HIGH"):
        parts.append(f"*{sev}*")
    elif cvss is not None:
        try:
            sc = float(cvss)
            if sc >= 9.0:
                parts.append("*CRITICAL*")
            elif sc >= 7.0:
                parts.append("*HIGH*")
            elif sc >= 4.0:
                parts.append("MEDIUM")
            else:
                parts.append("LOW")
        except Exception:
            pass

    # EPSS (score + optional percentile)
    epss = item.get("epss") or item.get("epss_score")
    epss_pct = (
        item.get("epss_percentile")
        or item.get("epss_pct")
        or item.get("epss_percent")
    )
    if epss is not None:
        try:
            label = f"EPSS {float(epss):.2f}"
            if epss_pct not in (None, ""):
                try:
                    # Accept 0–1 or 0–100 inputs
                    p = float(epss_pct)
                    if p <= 1.0:
                        p = round(p * 100)
                    else:
                        p = round(p)
                    label += f" ({int(p)}th pct)"
                except Exception:
                    pass
            parts.append(label)
        except Exception:
            pass

    # CISA KEV
    if item.get("kev") or item.get("known_exploited"):
        parts.append("KEV")

    return " • ".join(parts) if parts else ""

def _first_url(item):
    for k in ("advisory_url", "details_url", "vendor_url", "url", "link"):
        url = item.get(k)
        if url:
            return url
    return None

def _what_to_do_lines(item):
    for k in ("what_to_do", "actions", "mitigations"):
        v = item.get(k)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
        if isinstance(v, str) and v.strip():
            return [v.strip()]

    # Fallback actions (safe defaults)
    return [
        "Apply the vendor patch/update.",
        "Prioritize internet-exposed assets; verify with a scan.",
    ]

def _clean(text, limit=None):
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s

def render_item_text_core(item, idx, tone="simple"):
    """
    Returns Slack mrkdwn text for a single item.
    Expected keys (use what exists, gracefully handle missing):
      title, summary/content, severity/cvss, epss[/percentile], kev,
      advisory_url/details_url/url/link, what_to_do/actions/mitigations
    """
    title = _clean(item.get("title") or "Security update")
    badges = _fmt_badges(item)
    summary_src = item.get("summary") or item.get_
