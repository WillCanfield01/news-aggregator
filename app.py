import os
import time
import re
import feedparser
import hashlib
import threading
import torch
import torch.nn.functional as F
from html import unescape
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import openai

MAX_CACHED_ARTICLES = 300  # or 50, or however many you want to keep live

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Global Model Cache
_model = None
_tokenizer = None
_model_labels = None
_model_lock = threading.Lock()

# Preloaded Articles Cache
cached_articles = []

# Split feeds into two batches to stagger updates
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

current_batch_index = 0  # Used to rotate batches

FILTERED_CATEGORIES = set(["Unknown"])  # Keep only Unknown out

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

def strip_html(text):
    return unescape(re.sub(r"<[^>]+>", "", text))

def simple_summarize(text, max_words=50):
    words = strip_html(text).split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def summarize_with_openai(text):
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this article briefly and neutrally: {text}"}
            ],
            max_tokens=75,
            temperature=0.5,
            timeout=30
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI summarization failed:", e)
        return "Summary not available."

def generate_article_id(url, index):
    return f"article-{hashlib.md5(url.encode()).hexdigest()[:8]}-{index}"

def fetch_feed(url, use_ai=False):
    articles = []
    try:
        print(f"Fetching {url}...")
        feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
        print(f"Found {len(feed.entries)} entries in {url}")
        for index, entry in enumerate(feed.entries):
            title = entry.get("title", "No Title")
            desc = entry.get("summary", "")
            text = f"{title} {desc}"
            if not desc.strip():
                continue
            category = predict_category(text)
            if category != "Unknown":
                summary = summarize_with_openai(desc) if use_ai else simple_summarize(desc)
                # Extract source domain
                parsed_url = urlparse(url)
                source = parsed_url.netloc.replace("www.", "").replace("feeds.", "").split(".")[0].capitalize()
                articles.append({
                    "id": generate_article_id(url, index),
                    "title": title,
                    "summary": summary,
                    "description": desc,
                    "url": entry.get("link", "#"),
                    "category": category,
                    "source": source  # ✅ Add source here
                })

        print(f"✓ Added {len(articles)} articles from {url}")
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
    return articles

def preload_articles_batched(feed_list, use_ai=False):
    global cached_articles
    print(f"Preloading articles from {len(feed_list)} feeds...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda u: fetch_feed(u, use_ai), feed_list)
        new_articles = [article for feed in results for article in feed]

    # Filter duplicates
    existing_ids = {a["id"] for a in cached_articles}
    unique_new = [a for a in new_articles if a["id"] not in existing_ids]

    # Add new articles to the front (most recent first)
    cached_articles = unique_new + cached_articles

    # Trim to the latest MAX_CACHED_ARTICLES
    cached_articles = cached_articles[:MAX_CACHED_ARTICLES]

    print(f"✓ Total cached articles after trim: {len(cached_articles)}")

def periodic_refresh(interval=480):  # Every 8 minutes
    global current_batch_index
    while True:
        print(f"\n🌀 Refreshing batch {current_batch_index + 1}...")
        preload_articles_batched(RSS_FEED_BATCHES[current_batch_index], use_ai=False)
        current_batch_index = (current_batch_index + 1) % len(RSS_FEED_BATCHES)
        time.sleep(interval)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news")
def get_news():
    response = jsonify(cached_articles)
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route("/regenerate-summary/<article_id>")
def regenerate_summary(article_id):
    article = next((a for a in cached_articles if a["id"] == article_id), None)
    if article:
        article["summary"] = summarize_with_openai(article["description"])
        return jsonify({"summary": article["summary"]})
    return jsonify({"error": "Article not found"}), 404

@app.route("/refresh")
def manual_refresh():
    global current_batch_index
    print(f"🔁 Manual refresh via /refresh (batch {current_batch_index + 1})...")
    preload_articles_batched(RSS_FEED_BATCHES[current_batch_index], use_ai=False)
    current_batch_index = (current_batch_index + 1) % len(RSS_FEED_BATCHES)
    return jsonify({"status": "Refreshed", "batch": current_batch_index})

# Preload articles right before the app starts (safe across all Flask versions)
print("⚡ Preloading articles at startup...")
preload_articles_batched(RSS_FEED_BATCHES[0], use_ai=False)


if __name__ == "__main__":
    # Only in local/dev: safe to spawn background thread
    threading.Thread(target=periodic_refresh, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
