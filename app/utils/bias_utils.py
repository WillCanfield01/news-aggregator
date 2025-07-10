import re
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bias_cache = {}

def detect_political_bias(text, article_id=None, source=None):
    if article_id and article_id in bias_cache:
        return bias_cache[article_id]

    KNOWN_BIAS_BY_SOURCE = {
        "guardian": 20, "cnn": 35, "foxnews": 80, "nyt": 30,
        "reuters": 50, "npr": 45, "breitbart": 95,
    }
    fallback_bias = KNOWN_BIAS_BY_SOURCE.get((source or "").lower(), 50)

    prompt = (
        "Rate the political bias of this news article on a scale from 0 (Far Left), 50 (Center), to 100 (Far Right).\n\n"
        "Consider language, tone, and framing of issues. Even subtle preferences matter. Avoid assuming neutrality.\n\n"
        "Examples:\n"
        "- Praise of renewable energy and criticism of oil companies = 25\n"
        "- Defense of gun rights or religious liberty = 75\n"
        "- Objective economic stats without interpretation = 50\n"
        "- Article with loaded words like 'radical left' or 'MAGA patriots' = 10 or 95\n\n"
        "Return ONLY a number from 0 to 100."
    )

    try:
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Article: {text}"}
            ],
            max_tokens=10,
            temperature=0.3,
            timeout=10
        )
        raw = result.choices[0].message.content.strip()
        match = re.search(r'\d+', raw)
        if not match:
            print(f"Unexpected OpenAI output for bias: {raw}")
            return fallback_bias
        bias_score = int(match.group())

        if 45 <= bias_score <= 55 and source in KNOWN_BIAS_BY_SOURCE:
            delta = (KNOWN_BIAS_BY_SOURCE[source] - 50) * 0.3  # Apply 30% nudge
            bias_score += int(delta)
        bias_score = max(0, min(100, bias_score))
        if article_id:
            bias_cache[article_id] = bias_score
        return bias_score

    except Exception as e:
        print("Bias detection failed:", e)
        return fallback_bias

def bias_bucket(score):
    if score < 40:
        return "Left"
    elif score > 60:
        return "Right"
    else:
        return "Center"