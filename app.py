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
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import openai

MAX_CACHED_ARTICLES = 300

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = os.getenv("SECRET_KEY", "super-secret-dev-key")

@app.before_first_request
def create_tables():
    db.create_all()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CORS(app, supports_credentials=True)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(200))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

_model = None
_tokenizer = None
_model_labels = None
_model_lock = threading.Lock()

cached_articles = []
new_articles_last_refresh = []

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

current_batch_index = 0

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

def generate_article_id(link):
    return f"article-{hashlib.md5(link.encode()).hexdigest()[:12]}"

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
                parsed_url = urlparse(url)
                source = parsed_url.netloc.replace("www.", "").replace("feeds.", "").split(".")[0].capitalize()
                articles.append({
                    "id": generate_article_id(entry.get("link", f"{url}-{index}")),
                    "title": title,
                    "summary": summary,
                    "description": desc,
                    "url": entry.get("link", "#"),
                    "category": category,
                    "source": source
                })
        print(f"✓ Added {len(articles)} articles from {url}")
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
    return articles

def preload_articles_batched(feed_list, use_ai=False):
    global cached_articles, new_articles_last_refresh
    print(f"Preloading articles from {len(feed_list)} feeds...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda u: fetch_feed(u, use_ai), feed_list)
        new_articles = [article for feed in results for article in feed]
    existing_ids = {a["id"] for a in cached_articles}
    unique_new = [a for a in new_articles if a["id"] not in existing_ids]
    new_articles_last_refresh = unique_new
    cached_articles = unique_new + cached_articles
    cached_articles = cached_articles[:MAX_CACHED_ARTICLES]
    print(f"✓ {len(unique_new)} new articles added. Total cached: {len(cached_articles)}")

def periodic_refresh(interval=480):
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

@app.route("/new")
def get_new_articles():
    return jsonify(new_articles_last_refresh)

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()

        if not data or not data.get("username") or not data.get("password"):
            return jsonify({"error": "Username and password required"}), 400

        if User.query.filter_by(username=data["username"]).first():
            return jsonify({"error": "Username already exists"}), 400

        user = User(username=data["username"])
        user.set_password(data["password"])
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)
        return jsonify({"success": True, "message": "Signed up and logged in!"})
    except Exception as e:
        print("Signup error:", e)
        return jsonify({"error": "Server error", "details": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(username=data["username"]).first()
    if user and user.check_password(data["password"]):
        login_user(user, remember=True)
        return jsonify({"message": "Logged in successfully"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@app.route("/me")
@login_required
def me():
    return jsonify({"username": current_user.username})

print("⚡ Preloading articles at startup...")
preload_articles_batched(RSS_FEED_BATCHES[0], use_ai=False)

if __name__ == "__main__":
    # Run this only when directly launching app.py, not when importing
    threading.Thread(target=periodic_refresh, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

