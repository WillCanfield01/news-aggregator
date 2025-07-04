import os
import time
import re
import feedparser
import hashlib
import threading
import torch
import torch.nn.functional as F
import smtplib
import openai
import requests
import asyncio
import aiohttp
from html import unescape
from flask import Flask, jsonify, render_template, request, session, redirect, url_for, flash
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import quote_plus
from sqlalchemy.pool import QueuePool
from itsdangerous import URLSafeTimedSerializer
from email.mime.text import MIMEText
from postmarker.core import PostmarkClient
from functools import wraps
from datetime import datetime, timedelta
from newspaper import Article

MAX_CACHED_ARTICLES = 300
os.environ["TOKENIZERS_PARALLELISM"] = "false"
POSTMARK_TOKEN = os.getenv("POSTMARK_SERVER_TOKEN")
if not POSTMARK_TOKEN:
    raise RuntimeError("Missing POSTMARK_SERVER_TOKEN environment variable")

postmark = PostmarkClient(server_token=POSTMARK_TOKEN)

app = Flask(__name__)
uri = os.environ.get("DATABASE_URL", "")
if uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)
if "sslmode" not in uri:
    uri += "?sslmode=require"

app.config["SQLALCHEMY_DATABASE_URI"] = uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = os.getenv("SECRET_KEY", "super-secret-dev-key")

app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,     # Auto reconnect if connection drops
    "pool_recycle": 280,       # Close stale connections (Render may idle them)
    "pool_size": 5,
    "max_overflow": 10,
    "poolclass": QueuePool
}

app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bias_cache = {}
CORS(app, supports_credentials=True)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"  # if you have a login page route

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(200))
    saved_articles = db.relationship("SavedArticle", backref="user", lazy=True)
    zipcode = db.Column(db.String(10), nullable=True)

    # in your User model
    email = db.Column(db.String(120), unique=True, nullable=False)
    is_confirmed = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SavedArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    article_id = db.Column(db.String(64), nullable=False)
    title = db.Column(db.String(255))
    url = db.Column(db.String(500))
    summary = db.Column(db.Text)
    source = db.Column(db.String(100))
    category = db.Column(db.String(100))

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
        "https://rss.cnn.com/rss/edition.rss",
        "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best",
        "https://feeds.npr.org/1001/rss.xml",
    ],
    [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://www.theguardian.com/world/rss",
        "http://feeds.feedburner.com/TechCrunch/",
        "https://www.espn.com/espn/rss/news",
    ]
]

ZIP_RSS_MAP = {
    "10001": ["https://www.nydailynews.com/arcio/rss/category/news/?query=display_date:[now-7d+TO+now]"],
    "94105": ["https://www.sfchronicle.com/default/feed/local-news"],
    "90001": ["https://www.latimes.com/local/rss2.0.xml"],
}

CITY_RSS_MAP = {
    "Boise, ID": ["https://www.idahopress.com/rss/news"],
    "New York, NY": ["https://www.nydailynews.com/arcio/rss/category/news/?query=display_date:[now-7d+TO+now]"],
    "Los Angeles, CA": ["https://www.latimes.com/local/rss2.0.xml"],
    "San Francisco, CA": ["https://www.sfchronicle.com/default/feed/local-news"],
    "Chicago, IL": ["https://www.chicagotribune.com/arcio/rss/category/news/?query=display_date:[now-7d+TO+now]"]
}

DEFAULT_LOCAL_FEED = ["https://news.google.com/rss/search?q=local+news&hl=en-US&gl=US&ceid=US:en"]

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
                is_render = os.getenv("RENDER", "false").lower() == "true"
                model_path = "cardiffnlp/tweet-topic-21-multi" if is_render else "./models/cardiffnlp/tweet-topic-21-multi"
                _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=not is_render)
                _model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=not is_render).to("cpu").eval()
                config = AutoConfig.from_pretrained(model_path, local_files_only=not is_render)
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
            temperature=0.3,  # Lower temperature for more factual output
            timeout=30
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI summarization failed:", e)
        return "Summary not available."

def generate_article_id(link):
    return f"article-{hashlib.md5(link.encode()).hexdigest()[:12]}"

async def fetch_feed_async(session, url, use_ai=False, use_bias=True):
    articles = []
    bias = 50  # default fallback
    try:
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=aiohttp.ClientTimeout(total=10)) as response:
            content = await response.read()
            feed = feedparser.parse(content)

            cutoff = datetime.utcnow() - timedelta(days=7)
            parsed_url = urlparse(url)
            source = parsed_url.netloc.replace("www.", "").replace("feeds.", "").split(".")[0].lower()

            for index, entry in enumerate(feed.entries):
                title = entry.get("title", "No Title").strip()
                desc = entry.get("summary", "").strip() or entry.get("description", "").strip()
                link = entry.get("link", "").strip()

                if not title or not link:
                    continue

                parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
                pub_date = datetime(*parsed_date[:6]) if parsed_date else datetime.utcnow()
                if pub_date < cutoff:
                    continue

                text = f"{title} {desc}"
                category = predict_category(text)
                if category == "Unknown":
                    category = "General"

                summary = summarize_with_openai(desc or title) if use_ai else simple_summarize(desc or title)
                article_id = generate_article_id(link or f"{url}-{index}")
                if use_bias:
                    bias = detect_political_bias(f"{title}. {desc}", article_id=article_id, source=source)

                articles.append({
                    "id": article_id,
                    "title": title,
                    "summary": summary,
                    "description": desc,
                    "url": link,
                    "category": category,
                    "source": source,
                    "published": pub_date.isoformat(),
                    "published_dt": pub_date,
                    "bias": bias
                })

            articles.sort(key=lambda a: a.get("published_dt", datetime.min), reverse=True)
            print(f"✓ [Async] Added {len(articles)} from {url}")
    except Exception as e:
        print(f"⚠️ [Async] Failed to fetch {url}: {e}")
    return articles

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
        bias_score = int(re.search(r'\d+', raw).group())

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

async def preload_articles_batched_async(feed_list, use_ai=False, use_bias=True):
    global cached_articles, new_articles_last_refresh
    print(f"⚡ Async preloading from {len(feed_list)} feeds...")
    new_articles = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_feed_async(session, url, use_ai=use_ai, use_bias=use_bias)
            for url in feed_list
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            new_articles.extend(result)

    existing_ids = {a["id"] for a in cached_articles}
    unique_new = [a for a in new_articles if a["id"] not in existing_ids]
    new_articles_last_refresh = unique_new
    cached_articles = unique_new + cached_articles
    cached_articles = cached_articles[:MAX_CACHED_ARTICLES]
    print(f"✓ Async added {len(unique_new)} new articles. Total cached: {len(cached_articles)}")

async def periodic_refresh_async(interval=480):
    global current_batch_index
    while True:
        print(f"\n🌀 [Async] Refreshing batch {current_batch_index + 1}...")
        await preload_articles_batched_async(RSS_FEED_BATCHES[current_batch_index], use_ai=False)
        current_batch_index = (current_batch_index + 1) % len(RSS_FEED_BATCHES)
        await asyncio.sleep(interval)

def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(email, salt="email-confirmation-salt")

def confirm_token(token, expiration=3600):  # 1 hour
    serializer = URLSafeTimedSerializer(app.secret_key)
    try:
        email = serializer.loads(
            token,
            salt="email-confirmation-salt",
            max_age=expiration
        )
    except Exception:
        return False
    return email

def send_confirmation_email(email, username, token):
    confirm_link = f"https://therealroundup.com/confirm/{token}"
    postmark.emails.send(
        From=os.getenv("EMAIL_FROM"),
        To=email,
        Subject='Confirm Your Email – The Roundup',
        HtmlBody=f'''
            <p>Hi {username},</p>
            <p>Thanks for signing up! Please confirm your email by clicking the link below:</p>
            <p><a href="{confirm_link}">Confirm your email</a></p>
        ''',
        MessageStream="outbound"
    )

def serialize_article(article):
    return {
        "id": article.article_id,
        "title": article.title,
        "url": article.url,
        "summary": article.summary,
        "source": article.source,
        "category": article.category
    }

def extract_full_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"⚠️ Failed to extract article text: {e}")
        return ""

def resolve_zip_to_city(zipcode):
    try:
        response = requests.get(f"http://api.zippopotam.us/us/{zipcode}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data["places"][0]["place name"]
            state = data["places"][0]["state abbreviation"]
            return f"{city}, {state}"
    except Exception as e:
        print(f"Geolocation failed for zip {zipcode}: {e}")
    return None

zip_cache = {}

def resolve_zip_to_city_cached(zipcode):
    if zipcode in zip_cache:
        return zip_cache[zipcode]
    city = resolve_zip_to_city(zipcode)
    if city:
        zip_cache[zipcode] = city
    return city

local_feed_cache = {}

async def fetch_city_articles(city):
    if city in local_feed_cache and (datetime.utcnow() - local_feed_cache[city]["time"]).seconds < 600:
        return local_feed_cache[city]["data"]

    urls = CITY_RSS_MAP.get(city)
    if not urls:
        return []

    local_articles = []
    async with aiohttp.ClientSession() as session:
        for url in urls:
            articles = await fetch_feed_async(session, url, use_ai=False, use_bias=True)
            local_articles.extend(articles)

    local_feed_cache[city] = {
        "data": local_articles,
        "time": datetime.utcnow()
    }
    return local_articles

@app.route("/")
def home():
    return render_template("index.html")

import asyncio

@app.route("/news")
def get_news():
    articles = cached_articles
    if current_user.is_authenticated and current_user.zipcode:
        city = resolve_zip_to_city(current_user.zipcode)
        if city:
            local_articles = asyncio.run(fetch_city_articles(city))  # ⬅ Run the async method
            articles = local_articles + articles
    articles = articles[:MAX_CACHED_ARTICLES]
    return jsonify([{k: v for k, v in a.items() if k != "published_dt"} for a in articles])

@app.route("/regenerate-summary/<article_id>")
def regenerate_summary(article_id):
    article = next((a for a in cached_articles if a["id"] == article_id), None)
    if article:
        full_text = extract_full_article_text(article["url"])
        if not full_text:
            print(f"🟡 Falling back to RSS description for {article['url']}")
            full_text = article["description"]

        article["summary"] = summarize_with_openai(full_text)
        return jsonify({"summary": article["summary"]})
    return jsonify({"error": "Article not found"}), 404

@app.route("/refresh")
def manual_refresh():
    global current_batch_index
    print(f"🔁 Manual refresh via /refresh (batch {current_batch_index + 1})...")
    asyncio.run(preload_articles_batched_async(RSS_FEED_BATCHES[current_batch_index], use_ai=False))
    current_batch_index = (current_batch_index + 1) % len(RSS_FEED_BATCHES)
    return jsonify({"status": "Refreshed", "batch": current_batch_index})

@app.route("/new")
def get_new_articles():
    return jsonify(new_articles_last_refresh)

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}

    # Validate presence of username and password early
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    try:
        user = User.query.filter_by(username=username).first()
    except Exception as db_error:
        print("Database error during login:", db_error)
        return jsonify({"error": "Database error. Please try again shortly."}), 503  # ✅ Move return inside except
    if user and user.check_password(password):
        if not user.is_confirmed:
            return jsonify({"error": "Please confirm your email first."}), 403
        login_user(user)
        return jsonify({"success": True})
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

@app.route("/account")
@login_required
def account_page():
    saved = SavedArticle.query.filter_by(user_id=current_user.id).all()
    missing_zip = not current_user.zipcode
    return render_template(
        "account.html",
        username=current_user.username,
        saved_articles=saved,
        missing_zip=missing_zip,
        current_zip=current_user.zipcode  # Pass the current ZIP
    )

@app.route("/reset-password", methods=["POST"])
@login_required
def reset_password():
    data = request.get_json() or {}
    current = data.get("current_password", "").strip()
    new = data.get("new_password", "").strip()

    if not current_user.check_password(current):
        return jsonify({"error": "Current password is incorrect"}), 400

    if len(new) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400

    current_user.set_password(new)
    db.session.commit()
    return jsonify({"success": True, "message": "Password updated successfully"})

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip().lower()
    zipcode = data.get("zipcode", "").strip()
    user = User(username=username, email=email, zipcode=zipcode)

    if not username or not password or not email:
        return jsonify({"error": "All fields are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    try:
        user = User(username=username, email=email, zipcode=zipcode)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # ✅ These lines now run only on success
        token = generate_confirmation_token(user.email)
        send_confirmation_email(user.email, username, token)

        return jsonify({"success": True, "message": "Signup complete! Check your email to confirm."})
    except Exception as e:
        print("Signup error:", e)
        db.session.rollback()
        return jsonify({"error": "Signup failed. Please try again."}), 500
    
def build_google_local_rss(city_state):
    encoded = quote_plus(f"{city_state} local news")
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    
@app.route("/confirm/<token>")
def confirm_email(token):
    email = confirm_token(token)
    if not email:
        return "The confirmation link is invalid or has expired.", 400

    user = User.query.filter_by(email=email).first_or_404()

    if user.is_confirmed:
        return "Account already confirmed. Please login.", 200
    else:
        user.is_confirmed = True
        db.session.commit()
        return redirect("https://therealroundup.com/?confirmed=true")

@app.route("/save-article", methods=["POST"])
@login_required
def save_article():
    data = request.get_json() or {}
    article_id = data.get("id")
    title = data.get("title")
    url = data.get("url")
    summary = data.get("summary")
    source = data.get("source")
    category = data.get("category")

    if SavedArticle.query.filter_by(user_id=current_user.id, article_id=article_id).first():
        return jsonify({"error": "Article already saved"}), 400

    current_saved_count = SavedArticle.query.filter_by(user_id=current_user.id).count()
    if current_saved_count >= 10:
        return jsonify({"error": "Save limit reached (10 articles max). Please unsave one first."}), 403

    saved = SavedArticle(
        user_id=current_user.id,
        article_id=article_id,
        title=title,
        url=url,
        summary=summary,
        source=source,
        category=category
    )
    db.session.add(saved)
    db.session.commit()

    return jsonify({"success": True, "message": "Article saved"})

@login_required
def restricted():
    if not current_user.is_confirmed:
        return redirect(url_for('unconfirmed_notice'))
    # proceed normally

@app.route("/save", methods=["POST"])
@login_required
def alias_save_article():
    return save_article()

@app.route("/unsave-article", methods=["POST"])
@login_required
def unsave_article():
    data = request.get_json() or {}
    article_id = data.get("id")

    saved = SavedArticle.query.filter_by(user_id=current_user.id, article_id=article_id).first()
    if not saved:
        return jsonify({"error": "Article not found in saved list"}), 404

    db.session.delete(saved)
    db.session.commit()
    return jsonify({"success": True, "message": "Article unsaved"})

@app.route("/saved-articles")
@login_required
def saved_articles():
    saved = SavedArticle.query.filter_by(user_id=current_user.id).all()
    serialized = [{
        "id": article.article_id,
        "title": article.title,
        "url": article.url,
        "summary": article.summary,
        "source": article.source,
        "category": article.category
    } for article in saved]
    return jsonify(serialized)

def bias_bucket(score):
    if score < 40:
        return "Left"
    elif score > 60:
        return "Right"
    else:
        return "Center"

@app.route("/news/by-bias/<bias>")
def news_by_bias(bias):
    bias = bias.strip().capitalize()
    if bias not in {"Left", "Center", "Right"}:
        return jsonify({"error": "Invalid bias value"}), 400

    filtered = [
        a for a in cached_articles
        if bias_bucket(a["bias"]) == bias
    ]
    return jsonify(filtered)

@app.route("/news/by-category/<category>")
def news_by_category(category):
    normalized = normalize_category(category)
    filtered = [a for a in cached_articles if a["category"] == normalized]
    return jsonify(filtered)

@app.route("/unconfirmed")
def unconfirmed_notice():
    return render_template("unconfirmed.html"), 403

@app.route("/resend-confirmation", methods=["POST"])
@login_required
def resend_confirmation():
    if current_user.is_confirmed:
        return jsonify({"message": "Account already confirmed."}), 200
    token = generate_confirmation_token(current_user.email)
    send_confirmation_email(current_user.email, current_user.username, token)
    return jsonify({"message": "Confirmation email resent."}), 200

@app.route("/news/local")
@login_required
def local_news():
    zipcode = current_user.zipcode or "83646"
    city = resolve_zip_to_city_cached(zipcode)

    if city and city in CITY_RSS_MAP:
        feed_urls = CITY_RSS_MAP[city]
    else:
        feed_urls = DEFAULT_LOCAL_FEED

    articles = []
    for url in feed_urls:
        fetched = asyncio.run(fetch_from_single_url(url))
        articles.extend(fetched)

    articles.sort(key=lambda a: a.get("published_dt", datetime.min), reverse=True)
    return jsonify(articles[:10])

async def fetch_from_single_url(url):
    async with aiohttp.ClientSession() as session:
        return await fetch_feed_async(session, url, use_ai=True, use_bias=True)

from flask_login import login_required, current_user
from flask import redirect, url_for

@app.route("/local-news")
@login_required
def local_news_page():
    z = current_user.zipcode
    city = resolve_zip_to_city(z) if z else None
    if not city or city not in CITY_RSS_MAP:
        # Render page with fallback and show message instead of redirect
        flash("Local news requires a valid ZIP code in supported areas. Showing default local news feed.")
        return render_template("local_news.html", use_default_feed=True)
    return render_template("local_news.html", city=city, use_default_feed=False)

@app.route("/update-zipcode", methods=["POST"])
@login_required
def update_zipcode():
    zipcode = request.form.get("zipcode", "").strip()

    if zipcode and zipcode.isdigit() and 4 <= len(zipcode) <= 10:
        city = resolve_zip_to_city(zipcode)
        if city and city in CITY_RSS_MAP:
            current_user.zipcode = zipcode
            db.session.commit()
            flash("ZIP code updated!")
        else:
            flash("ZIP code not supported for local news. Please enter a different ZIP.")
    else:
        flash("Invalid ZIP code. Please enter a valid number.")

    return redirect(url_for("account_page"))

@login_manager.unauthorized_handler
def unauthorized():
    # Redirect unauthorized users to the login page
    return redirect(url_for("login"))

def start_background_tasks():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(preload_articles_batched_async(RSS_FEED_BATCHES[0], use_ai=False))
    loop.create_task(periodic_refresh_async())
    loop.run_forever()

if __name__ == "__main__":
    threading.Thread(target=start_background_tasks, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))