import os
import requests
import torch
import time
import torch.nn.functional as F
import feedparser
import re
import threading
from html import unescape
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import openai
import hashlib

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Global lazy singleton objects
_model = None
_tokenizer = None
_model_labels = None
_model_lock = threading.Lock()


def get_model_components():
    global _model, _tokenizer, _model_labels
    if _model is None or _tokenizer is None or _model_labels is None:
        with _model_lock:
            if _model is None or _tokenizer is None or _model_labels is None:
                try:
                    print("DEBUG: Lazy loading model and tokenizer...")
                    _tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
                    _model = AutoModelForSequenceClassification.from_pretrained(
                        "cardiffnlp/tweet-topic-21-multi"
                    ).to("cpu").eval()
                    config = AutoConfig.from_pretrained("cardiffnlp/tweet-topic-21-multi")
                    _model_labels = list(config.id2label.values())
                    print("DEBUG: Model loaded with labels:", _model_labels)
                except Exception as e:
                    print(f"ERROR loading model: {e}")
                    raise RuntimeError("Model failed to load")
    return _model, _tokenizer, _model_labels


RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/edition.rss",
    "http://feeds.reuters.com/reuters/topNews",
    "https://feeds.npr.org/1001/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.theguardian.com/world/rss",
    "http://feeds.feedburner.com/TechCrunch/",
    "https://www.espn.com/espn/rss/news",
]

CATEGORY_MAP = {
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

FILTERED_CATEGORIES = set(CATEGORY_MAP.keys())


def normalize_category(category):
    key = category.lower().replace(" ", "_")
    return CATEGORY_MAP.get(key, category.title())


def predict_category(article_text, confidence_threshold=0.5):
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


def strip_html(text):
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean)


def simple_summarize(text, max_words=50):
    if not text:
        return "No description available."
    clean_text = strip_html(text)
    words = clean_text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def summarize_with_openai(article_text):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following article, and make sure the summary is easy to understand, also make sure the results sound human and remain politically neutral: {article_text}"}
            ],
            max_tokens=75,
            temperature=0.5,
            timeout=30
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Failed to summarize article."


articles = []


def generate_article_id(url, index):
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"article-{url_hash}-{index}"


def fetch_news_from_rss(use_ai=False):
    print("DEBUG: Fetching news from RSS feeds...")
    global articles
    all_articles = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            print(f"DEBUG: {len(feed.entries)} entries from {url}")
            for index, entry in enumerate(feed.entries[:10]):
                title = entry.get("title", "No Title")
                description = entry.get("summary", "")
                link = entry.get("link", "#")
                text = f"{title} {description}"

                if description.strip():
                    category = predict_category(text)
                    if category.lower().replace(" ", "_") in FILTERED_CATEGORIES:
                        summary = summarize_with_openai(description) if use_ai else simple_summarize(description)
                        all_articles.append({
                            "id": generate_article_id(url, index),
                            "title": title,
                            "summary": summary,
                            "description": description,
                            "url": link,
                            "category": category
                        })
        except Exception as e:
            print(f"ERROR parsing RSS feed {url}: {e}")

    print(f"DEBUG: Total articles from RSS: {len(all_articles)}")
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


@app.route("/regenerate-summary/<article_id>", methods=["GET"])
def regenerate_summary(article_id):
    print(f"DEBUG: Received request to regenerate summary for {article_id}")
    article = next((a for a in articles if a['id'] == article_id), None)
    if article is None:
        return jsonify({"error": "Article not found"}), 404
    new_summary = summarize_with_openai(article["description"])
    return jsonify({"summary": new_summary})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
