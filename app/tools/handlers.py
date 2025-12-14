from __future__ import annotations

import os
from typing import Any, Dict

import openai

# Reuse existing OpenAI integration style
_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_DEFAULT_MODEL = "gpt-4.1-mini"


def run_resume_bullets(payload: Dict[str, Any]) -> Dict[str, str]:
    """Generate improved resume bullets."""
    experience_text = (payload.get("experience_text") or "").strip()
    target_role = (payload.get("target_role") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    tone = (payload.get("tone") or "Professional").strip() or "Professional"

    prompt = (
        "You are a concise resume assistant. Improve the provided experience notes into numbered bullets.\n"
        "- Return 6-10 numbered bullets only (e.g., '1. ...').\n"
        "- Use strong action verbs, keep one line each.\n"
        "- Add realistic metric placeholders if missing (e.g., 'reduced by X%').\n"
        "- Align with the target role and any job description keywords without inventing credentials.\n"
        "- Prefer IT support, security, automation, and troubleshooting emphasis if the target role suggests it.\n"
        f"Tone: {tone}\n"
        f"Target role: {target_role or 'General'}\n"
        f"Job description (optional): {job_description or 'N/A'}\n"
        f"Experience notes: {experience_text}\n"
        "Output only the bullets."
    )

    try:
        result = _client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You improve resume bullets with brevity and measurable impact."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=650,
            temperature=0.2,
            timeout=15,
        )
        content = result.choices[0].message.content.strip()
        return {"output": content}
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc
