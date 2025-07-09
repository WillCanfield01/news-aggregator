import re
import feedparser
import hashlib
import torch
import torch.nn.functional as F
from urllib.parse import urlparse
from datetime import datetime, timedelta
from html import unescape
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from app.utils.bias_utils import detect_political_bias
from openai import OpenAI
from newspaper import Article

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_model = None
_tokenizer = None
_model_labels = None

def get_model_components():
    global _model, _tokenizer, _model_labels
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
        _model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi").to("cpu").eval()
        config = AutoConfig.from_pretrained("cardiffnlp/tweet-topic-21-multi")
        _model_labels = list(config.id2label.values())
    return _model, _tokenizer, _model_labels

CATEGORY_MAP = {
    "arts_&_culture": "Entertainment", "fashion_&_style": "Entertainment",
    "food_&_dining": "Entertainment", "diaries_&_daily_life": "Entertainment",
    "business_&_entrepreneurs": "Finance", "science_&_technology": "Technology",
    "sports": "Sports", "health": "Health", "politics": "Politics",
    "news_&_social_concern": "Politics", "other_hobbies": "Entertainment",
    "music": "Entertainment", "travel_&_adventure": "Entertainment",
    "celebrity_&_pop_culture": "Entertainment", "gaming": "Entertainment",
    "learning_&_educational": "Education", "fitness_&_health": "Health",
    "youth_&_student_life": "Education", "relationships": "Lifestyle",
    "family": "Lifestyle"
}

def normalize_category(category):
    return CATEGORY_MAP.get(category.lower().replace(" ", "_"), category.title())

def predict_category(text):
    if not text.strip():
        return "Unknown"
    try:
        model, tokenizer, labels = get_model_components()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        idx = torch.argmax(probs).item()
        conf = probs[0][idx].item()
        if idx >= len(labels) or conf < 0.5:
            return "Unknown"
        raw = labels[idx].lower().replace(" ", "_")
        return normalize_category(raw)
    except:
        return "Unknown"

def generate_article_id(link):
    return f"article-{hashlib.md5(link.encode()).hexdigest()[:12]}"

def strip_html(text):
    return unescape(re.sub(r"<[^>]+>", "", text))

def simple_summarize(text, max_words=50):
    words = strip_html(text).split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def summarize_with_openai(text):
    try:
        text = text[:12000]
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a news summarization assistant. Generate a concise, factual summary. "
                    "Avoid bias, speculation, or exaggeration.")},
                {"role": "user", "content": f"Summarize in 2-3 sentences:\n\n{text}"}
            ],
            max_tokens=180,
            temperature=0.3,
            timeout=30
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print("Summarization failed:", e)
        return "Summary not available."

def extract_full_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print("Full article extraction failed:", e)
        return ""