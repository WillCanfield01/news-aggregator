import re
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bias_cache = {}

KNOWN_BIAS_BY_SOURCE = {
    "guardian": 20, "cnn": 35, "foxnews": 80, "nyt": 30,
    "reuters": 50, "npr": 45, "breitbart": 95,
}

def detect_political_bias(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a political bias scoring assistant. Given a news headline or short article text, "
                    "respond with ONLY a single integer from -100 to 100, where:\n"
                    "-100 means strongly left-leaning,\n"
                    "0 means neutral,\n"
                    "100 means strongly right-leaning.\n"
                    "Do NOT explain your answer or include any other text. Reply with just the number."
                )},
                {"role": "user", "content": f"Score the political bias of this text:\n\n{text}"}
            ],
            max_tokens=5,
            temperature=0.3,
            timeout=15
        )
        score_str = response.choices[0].message.content.strip()
        return int(score_str)
    except Exception as e:
        print(f"Bias detection failed: {e}")
        return 50  # fallback to neutral

def bias_bucket(score):
    if score < 40:
        return "Left"
    elif score > 60:
        return "Right"
    return "Center"