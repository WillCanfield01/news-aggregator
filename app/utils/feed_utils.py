import os
import re
import hashlib
import feedparser
import torch
import torch.nn.functional as F
import threading
import asyncio
from urllib.parse import urlparse, quote_plus
from datetime import datetime, timedelta
from html import unescape
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from app.utils.bias_utils import detect_political_bias
from openai import OpenAI
from newspaper import Article
from app.utils.geo_utils import get_city_state_from_zip, make_local_news_query
import pgeocode

# For local feed city/state lookups
nomi = pgeocode.Nominatim('us')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Article batching & caching ===
MAX_CACHED_ARTICLES = 300
cached_articles = []
new_articles_last_refresh = []
current_batch_index = 0

RSS_FEED_BATCHES = [
    [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://rss.cnn.com/rss/edition.rss",
        "http://feeds.reuters.com/reuters/topNews",
        "https://feeds.npr.org/1001/rss.xml",
    ],
    [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://www.theguardian.com/world/rss",
        "http://feeds.feedburner.com/TechCrunch/",
        "https://www.espn.com/espn/rss/news",
    ]
]

FILTERED_CATEGORIES = set(["Unknown"])
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

_model = None
_tokenizer = None
_model_labels = None
_model_lock = threading.Lock()

# =============================
# ML/UTILS: Category + Summary
# =============================
def get_model_components():
    global _model, _tokenizer, _model_labels
    if _model is None:
        with _model_lock:
            if _model is None:
                _tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
                _model = AutoModelForSequenceClassification.from_pretrained(
                    "cardiffnlp/tweet-topic-21-multi"
                ).to("cpu").eval()
                config = AutoConfig.from_pretrained("cardiffnlp/tweet-topic-21-multi")
                _model_labels = list(config.id2label.values())
    return _model, _tokenizer, _model_labels

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
        # Limit to around 3,000 tokens worth of text (OpenAI estimates 4 tokens per word avg)
        text = text[:12000]  # roughly 3000 tokens (~9000-12000 chars)
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a news summarization assistant. "
                        "Your task is to generate a concise, fact-based summary of a full news article. "
                        "Avoid exaggeration, emotional language, or political bias. "
                        "Remain strictly neutral, objective, and accurate. Do not speculate or editorialize."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following news article in 2-3 sentences. "
                        f"Focus only on the main facts and events without inserting opinion or bias:\n\n{text}"
                    )
                }
            ],
            max_tokens=180,  # Enough for 2–3 sentence summary
            temperature=0.3,
            timeout=30
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI summarization failed:", e)
        return "Summary not available."

def extract_full_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"⚠️ Failed to extract article text: {e}")
        return ""

# ================
# Main Feed Logic
# ================
def fetch_feed(url, use_ai=False):
    articles = []
    try:
        print(f"Fetching {url}…")
        feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
        cutoff = datetime.utcnow() - timedelta(days=7)

        for index, entry in enumerate(feed.entries):
            title = entry.get("title", "No Title")
            desc  = entry.get("summary", "")
            if not desc.strip():
                continue

            # Parse publish date (allow updated if published missing)
            parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
            if not parsed_date:
                continue
            pub_date = datetime(*parsed_date[:6])
            if pub_date < cutoff:
                continue

            # Category & summary
            text     = f"{title} {desc}"
            category = predict_category(text)
            if category == "Unknown":
                category = "General"
            summary = summarize_with_openai(desc) if use_ai else simple_summarize(desc)

            # Source normalization
            parsed_url = urlparse(url)
            source     = parsed_url.netloc.replace("www.", "").replace("feeds.", "").split(".")[0].lower()

            # Political bias (internal fallback inside detect_political_bias)
            article_id = generate_article_id(entry.get("link", f"{url}-{index}"))
            bias = detect_political_bias(f"{title}. {desc}", article_id=article_id, source=source)

            articles.append({
                "id":          article_id,
                "title":       title,
                "summary":     summary,
                "description": desc,
                "url":         entry.get("link", "#"),
                "category":    category,
                "source":      source,
                "published":   pub_date.isoformat(),
                "bias":        bias
            })

        # Sort newest first
        articles.sort(key=lambda a: a["published"], reverse=True)
        print(f"✓ Added {len(articles)} articles from {url}")

    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")

    return articles

def preload_articles_batched(feed_list, use_ai=False):
    global cached_articles, new_articles_last_refresh
    print(f"Preloading articles from {len(feed_list)} feeds...")
    all_new = []
    for url in feed_list:
        all_new.extend(fetch_feed(url, use_ai))
    # Only keep unique articles
    existing_ids = {a["id"] for a in cached_articles}
    unique_new = [a for a in all_new if a["id"] not in existing_ids]
    new_articles_last_refresh = unique_new
    cached_articles = unique_new + cached_articles
    cached_articles = cached_articles[:MAX_CACHED_ARTICLES]
    print(f"✓ {len(unique_new)} new articles added. Total cached: {len(cached_articles)}")

def manual_refresh_articles():
    global current_batch_index
    preload_articles_batched(RSS_FEED_BATCHES[current_batch_index], use_ai=False)
    current_batch_index = (current_batch_index + 1) % len(RSS_FEED_BATCHES)
    return {"status": "Refreshed", "batch": current_batch_index}

def get_cached_articles():
    return cached_articles

def get_new_articles():
    return new_articles_last_refresh

def regenerate_summary_for_article(article_id):
    article = next((a for a in cached_articles if a["id"] == article_id), None)
    if article:
        full_text = extract_full_article_text(article["url"])
        if not full_text:
            full_text = article["description"]
        article["summary"] = summarize_with_openai(full_text)
        return article["summary"]
    return None

# ================
# Local News Logic
# ================
def is_valid_zip(zipcode):
    return bool(re.match(r"^\d{5}$", zipcode))

async def fetch_google_local_feed(zipcode: str, limit: int = 50):
    query = make_local_news_query(zipcode)
    if not query:
        print(f"⚠️ Could not resolve ZIP {zipcode} to city/state")
        return []

    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    print(f"📡 Fetching Google News RSS for: {query}")
    # If you want this to be async, implement an async fetch_single_feed
    return fetch_feed(url)[:limit]

def get_local_articles_for_user(user):
    if not user or not is_valid_zip(user.zipcode):
        return []
    # For parity, always fetch live—optionally, implement caching if needed
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    articles = loop.run_until_complete(fetch_google_local_feed(user.zipcode, limit=50))
    loop.close()
    return articles