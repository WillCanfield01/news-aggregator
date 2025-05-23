import os
import hashlib
import re
import threading
from html import unescape
from typing import List, Dict
import concurrent.futures
import time

import torch
import torch.nn.functional as F
import feedparser
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

_model = None
_tokenizer = None
_model_labels = None
_model_lock = threading.Lock()

def get_model_components():
    global _model, _tokenizer, _model_labels
    if _model is None or _tokenizer is None or _model_labels is None:
        with _model_lock:
            if _model is None or _tokenizer is None or _model_labels is None:
                print("DEBUG: Loading model and tokenizer...")
                _tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
                _model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/tweet-topic-21-multi").to("cpu").eval()
                config = AutoConfig.from_pretrained("cardiffnlp/tweet-topic-21-multi")
                _model_labels = list(config.id2label.values())
                print(f"DEBUG: Model loaded with labels: {_model_labels}")
    return _model, _tokenizer, _model_labels

RSS_FEEDS: List[str] = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/edition.rss",
    "http://feeds.reuters.com/reuters/topNews",
    "https://feeds.npr.org/1001/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.theguardian.com/world/rss",
    "http://feeds.feedburner.com/TechCrunch/",
    "https://www.espn.com/espn/rss/news",
]

FILTERED_CATEGORIES = {
    "Arts_&_culture", "Business_&_entrepreneurs", "Celebrity_&_pop_culture",
    "Diaries_&_daily_life", "Family", "Fashion_&_style", "Film_tv_&_video",
    "Fitness_&_health", "Food_&_dining", "Gaming", "Learning_&_educational",
    "Music", "News_&_social_concern", "Other_hobbies", "Relationships",
    "Science_&_technology", "Sports", "Travel_&_adventure", "Youth_&_student_life",
    "Entertainment", "Health", "Politics", "Finance", "Technology", "Unknown"
}

CATEGORY_MAP = {
    "Arts_&_culture": "Entertainment",
    "Fashion_&_style": "Entertainment",
    "Food_&_dining": "Entertainment",
    "Diaries_&_daily_life": "Entertainment",
    "Business_&_entrepreneurs": "Finance",
    "Science_&_technology": "Technology",
    "Sports": "Sports",
    "Health": "Health",
    "Politics": "Politics",
    "News_&_social_concern": "Politics",
    "Other_hobbies": "Entertainment",
    "Music": "Entertainment",
    "Travel_&_adventure": "Entertainment",
    "Celebrity_&_pop_culture": "Entertainment",
    "Gaming": "Entertainment",
    "Learning_&_educational": "Education",
    "Fitness_&_health": "Health",
    "Youth_&_student_life": "Education",
    "Relationships": "Lifestyle",
    "Family": "Lifestyle"
}

def normalize_category(category: str) -> str:
    category_map = {
        "arts_&_culture": "Entertainment",
        "business_&_entrepreneurs": "Finance",
        "celebrity_&_pop_culture": "Entertainment",
        "diaries_&_daily_life": "Entertainment",
        "family": "Lifestyle",
        "fashion_&_style": "Entertainment",
        "film_tv_&_video": "Entertainment",
        "fitness_&_health": "Health",
        "food_&_dining": "Entertainment",
        "gaming": "Entertainment",
        "learning_&_educational": "Education",
        "music": "Entertainment",
        "news_&_social_concern": "Politics",
        "other_hobbies": "Entertainment",
        "relationships": "Lifestyle",
        "science_&_technology": "Technology",
        "sports": "Sports",
        "travel_&_adventure": "Entertainment",
        "youth_&_student_life": "Education"
    }
    key = category.lower().replace(" ", "_")
    return category_map.get(key, category.title())

def predict_category(article_text: str, confidence_threshold: float = 0.5) -> str:
    if not article_text.strip():
        return "Unknown"
    try:
        model, tokenizer, labels = get_model_components()
        inputs = tokenizer(article_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

        if predicted_index >= len(labels):
            return "Unknown"

        raw_label = labels[predicted_index].lower().replace(" ", "_")
        normalized_category = normalize_category(raw_label)

        return normalized_category if confidence >= confidence_threshold else "Unknown"
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return "Unknown"

def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean)

def simple_summarize(text: str, max_words: int = 50) -> str:
    if not text:
        return "No description available."
    clean_text = strip_html(text)
    words = clean_text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

summary_cache = {}
summary_cache_lock = threading.Lock()
SUMMARY_CACHE_TTL = 3600  # Cache expiry in seconds

def get_cached_summary(article_id: str) -> str:
    with summary_cache_lock:
        cached = summary_cache.get(article_id)
        if cached and (time.time() - cached['time'] < SUMMARY_CACHE_TTL):
            return cached['summary']
        return None

def set_cached_summary(article_id: str, summary: str):
    with summary_cache_lock:
        summary_cache[article_id] = {'summary': summary, 'time': time.time()}

def summarize_with_openai(article_text: str) -> str:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Summarize the following article, and make sure the summary is easy to understand, "
                        "also make sure the results sound human and remain politically neutral: "
                        f"{article_text}"
                    ),
                }
            ],
            max_tokens=75,
            temperature=0.5,
            timeout=30
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI summarization: {e}")
        return "Failed to summarize article."

def generate_article_id(url: str, index: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"article-{url_hash}-{index}"

def fetch_single_feed(url: str, max_articles_per_feed=10, use_ai=False) -> List[Dict]:
    articles = []
    try:
        feed = feedparser.parse(url)
        print(f"DEBUG: Fetched {len(feed.entries)} entries from {url}")
        for idx, entry in enumerate(feed.entries[:max_articles_per_feed]):
            title = entry.get("title", "No Title")
            description = entry.get("summary", "")
            link = entry.get("link", "#")
            combined_text = f"{title} {description}"

            if description.strip():
                category = predict_category(combined_text)
                if category in FILTERED_CATEGORIES:
                    article_id = generate_article_id(url, idx)
                    summary = get_cached_summary(article_id)
                    if summary is None:
                        # Use quick simple summary for initial fetch
                        summary = simple_summarize(description)
                        if use_ai:
                            # Asynchronously generate AI summary (here we just set a placeholder to regenerate later)
                            threading.Thread(target=regenerate_summary_background, args=(article_id, description)).start()
                        set_cached_summary(article_id, summary)

                    articles.append({
                        "id": article_id,
                        "title": title,
                        "summary": summary,
                        "description": description,
                        "url": link,
                        "category": category
                    })
    except Exception as e:
        print(f"ERROR parsing RSS feed {url}: {e}")
    return articles

def regenerate_summary_background(article_id: str, description: str):
    summary = summarize_with_openai(description)
    set_cached_summary(article_id, summary)
    print(f"DEBUG: Background summary regenerated for {article_id}")

def fetch_news_from_rss(use_ai: bool = False) -> List[Dict]:
    print("DEBUG: Fetching news concurrently from RSS feeds...")
    all_articles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_single_feed, url, 10, use_ai) for url in RSS_FEEDS]
        for future in concurrent.futures.as_completed(futures):
            all_articles.extend(future.result())
    print(f"DEBUG: Total articles collected: {len(all_articles)}")
    global articles
    articles = all_articles
    return all_articles

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news")
def get_news():
    use_ai = request.args.get("ai", "0") == "1"
    global articles
    articles = fetch_news_from_rss(use_ai=use_ai)
    return jsonify(articles)

@app.route("/article-ids")
def article_ids():
    return jsonify([a['id'] for a in articles])

@app.route("/regenerate-summary/<article_id>", methods=["GET"])
def regenerate_summary(article_id: str):
    print(f"DEBUG: Regenerate summary requested for {article_id}")
    article = next((a for a in articles if a['id'] == article_id), None)
    if article is None:
        return jsonify({"error": "Article not found"}), 404
    new_summary = summarize_with_openai(article["description"])
    set_cached_summary(article_id, new_summary)
    return jsonify({"summary": new_summary})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
