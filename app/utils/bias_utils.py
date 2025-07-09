import re
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bias_cache = {}

KNOWN_BIAS_BY_SOURCE = {
    "guardian": 20, "cnn": 35, "foxnews": 80, "nyt": 30,
    "reuters": 50, "npr": 45, "breitbart": 95,
}

def detect_political_bias(text, article_id=None, source=None):
    if article_id and article_id in bias_cache:
        return bias_cache[article_id]

    fallback_bias = KNOWN_BIAS_BY_SOURCE.get((source or "").lower(), 50)

    try:
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "Rate political bias from 0 (far left) to 100 (far right). Use linguistic cues.")},
                {"role": "user", "content": f"Article: {text}"}
            ],
            max_tokens=10,
            temperature=0.3,
            timeout=10
        )
        raw = result.choices[0].message.content.strip()
        match = re.search(r'\d+', raw)
        if match:
            bias_score = int(match.group())
        else:
            raise ValueError(f"Could not parse bias score from: {raw}")

        if 45 <= bias_score <= 55 and source in KNOWN_BIAS_BY_SOURCE:
            delta = (KNOWN_BIAS_BY_SOURCE[source] - 50) * 0.3
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
    return "Center"