import os
import time
import re
import feedparser
import hashlib
import threading
import torch
import torch.nn.functional as F
import openai
import aiohttp
import asyncio
import pgeocode
import pandas as pd
from html import unescape
from flask import Blueprint, jsonify, render_template, request, redirect, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from postmarker.core import PostmarkClient
from datetime import datetime, timedelta
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, quote_plus

# Assuming db and login_manager are created elsewhere and imported into your app context!
from app import db, login_manager

# --- Blueprint registration ---
aggregator_bp = Blueprint(
    "aggregator", __name__,
    template_folder="templates",
    static_folder="static"
)

# --- Global setup ---
MAX_CACHED_ARTICLES = 300
os.environ["TOKENIZERS_PARALLELISM"] = "false"
POSTMARK_TOKEN = os.getenv("POSTMARK_SERVER_TOKEN")
postmark = PostmarkClient(server_token=POSTMARK_TOKEN)
local_articles_cache = {}
local_cache_lock = threading.Lock()
nomi = pgeocode.Nominatim('us')
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bias_cache = {}

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
        "https://reutersbest.com/feed/",
        "https://feeds.npr.org/1001/rss.xml",
        "https://news.google.com/rss/search?q=when:24h allinurl:apnews.com",
        "https://moxie.foxnews.com/google-publisher/latest.xml",
    ],
    [
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://www.theguardian.com/world/rss",
        "https://www.espn.com/espn/rss/news",
        "https://www.msnbc.com/feeds/latest",
        "https://feeds.bloomberg.com/politics/news.rss",
        "https://nypost.com/feed/",
    ]
]
current_batch_index = 0

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

# --- Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(200))
    saved_articles = db.relationship("SavedArticle", backref="user", lazy=True)
    zipcode = db.Column(db.String(10))
    email = db.Column(db.String(120), unique=True, nullable=False)
    is_confirmed = db.Column(db.Boolean, default=False)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

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
def load_user(user_id): return User.query.get(int(user_id))

# --- Utility functions ---
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
    if not text.strip(): return "Unknown"
    try:
        model, tokenizer, labels = get_model_components()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        idx = torch.argmax(probs).item()
        conf = probs[0][idx].item()
        if idx >= len(labels) or conf < 0.5: return "Unknown"
        raw = labels[idx].lower().replace(" ", "_")
        return normalize_category(raw)
    except: return "Unknown"

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
                {"role": "system", "content":
                    "You are a news summarization assistant. "
                    "Your task is to generate a concise, fact-based summary of a full news article. "
                    "Avoid exaggeration, emotional language, or political bias. "
                    "Remain strictly neutral, objective, and accurate. Do not speculate or editorialize."},
                {"role": "user", "content":
                    f"Summarize the following news article in 2-3 sentences. Focus only on the main facts and events without inserting opinion or bias:\n\n{text}"}
            ],
            max_tokens=180,
            temperature=0.3,
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
        print(f"Fetching {url}…")
        feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
        cutoff = datetime.utcnow() - timedelta(days=7)
        for index, entry in enumerate(feed.entries):
            title = entry.get("title", "No Title")
            desc  = entry.get("summary", "")
            if not desc.strip():
                continue
            parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
            if not parsed_date: continue
            pub_date = datetime(*parsed_date[:6])
            if pub_date < cutoff: continue
            text = f"{title} {desc}"
            category = predict_category(text)
            if category == "Unknown": category = "General"
            summary = summarize_with_openai(desc) if use_ai else simple_summarize(desc)
            parsed_url = urlparse(url)
            source = parsed_url.netloc.replace("www.", "").replace("feeds.", "").split(".")[0].lower()
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
        articles.sort(key=lambda a: a["published"], reverse=True)
        print(f"✓ Added {len(articles)} articles from {url}")
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
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
            delta = (KNOWN_BIAS_BY_SOURCE[source] - 50) * 0.3
            bias_score += int(delta)
        bias_score = max(0, min(100, bias_score))
        if article_id:
            bias_cache[article_id] = bias_score
        return bias_score
    except Exception as e:
        print("Bias detection failed:", e)
        return fallback_bias

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

# --- ROUTES ---
@aggregator_bp.route("/news")
def news_home():
    return render_template("aggregator_index.html")

@aggregator_bp.route("/api/news")
def get_news():
    response = jsonify(cached_articles)
    response.headers['Cache-Control'] = 'no-store'
    return response

@aggregator_bp.route("/refresh")
def manual_refresh():
    global current_batch_index
    print(f"🔁 Manual refresh via /refresh (batch {current_batch_index + 1})...")
    preload_articles_batched(RSS_FEED_BATCHES[current_batch_index], use_ai=False)
    current_batch_index = (current_batch_index + 1) % len(RSS_FEED_BATCHES)
    return jsonify({"status": "Refreshed", "batch": current_batch_index})

@aggregator_bp.route("/new")
def get_new_articles():
    return jsonify(new_articles_last_refresh)

@aggregator_bp.route("/regenerate-summary/<article_id>")
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

def extract_full_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"⚠️ Failed to extract article text: {e}")
        return ""

@aggregator_bp.route("/news/by-bias/<bias>")
def news_by_bias(bias):
    def bias_bucket(score):
        if score < 40:
            return "Left"
        elif score > 60:
            return "Right"
        else:
            return "Center"
    bias = bias.strip().capitalize()
    if bias not in {"Left", "Center", "Right"}:
        return jsonify({"error": "Invalid bias value"}), 400
    filtered = [a for a in cached_articles if bias_bucket(a["bias"]) == bias]
    return jsonify(filtered)

@aggregator_bp.route("/news/by-category/<category>")
def news_by_category(category):
    normalized = normalize_category(category)
    filtered = [a for a in cached_articles if a["category"] == normalized]
    return jsonify(filtered)

# Optional: Catchall for 404s under this blueprint
@aggregator_bp.route("/<path:invalid_path>")
def trap_invalid(invalid_path):
    if "wp-" in invalid_path:
        return "", 204
    return "Not Found", 404

# --- Background task starter (import and call from app.py) ---
def start_background_tasks():
    threading.Thread(target=periodic_refresh, daemon=True).start()
    # If you want to add local news background refresh, add here!
# ...[ALL YOUR CODE ABOVE]...

# ----- USER AUTH AND ACCOUNT ROUTES -----
@aggregator_bp.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip().lower()
    if not username or not password or not email:
        return jsonify({"error": "All fields are required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400
    try:
        zipcode_input = data.get("zipcode", "").strip()
        user = User(username=username, email=email, zipcode=zipcode_input)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        # Email confirmation
        token = generate_confirmation_token(user.email)
        send_confirmation_email(user.email, username, token)
        return jsonify({"success": True, "message": "Signup complete! Check your email to confirm."})
    except Exception as e:
        print("Signup error:", e)
        db.session.rollback()
        return jsonify({"error": "Signup failed. Please try again."}), 500

@aggregator_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    try:
        user = User.query.filter_by(username=username).first()
    except Exception as db_error:
        print("Database error during login:", db_error)
        return jsonify({"error": "Database error. Please try again shortly."}), 503
    if user and user.check_password(password):
        if not user.is_confirmed:
            return jsonify({"error": "Please confirm your email first."}), 403
        login_user(user)
        return jsonify(success=True, username=user.username)
    else:
        return jsonify(success=False, message="Invalid credentials"), 401

@aggregator_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@aggregator_bp.route("/me")
@login_required
def me():
    return jsonify({"username": current_user.username})

@aggregator_bp.route("/account")
@login_required
def account_page():
    saved = SavedArticle.query.filter_by(user_id=current_user.id).all()
    return render_template("account.html", username=current_user.username, saved_articles=saved)

@aggregator_bp.route("/reset-password", methods=["POST"])
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

@aggregator_bp.route("/confirm/<token>")
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
        return redirect(url_for("aggregator.home"))  # or custom URL

def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY", "super-secret-dev-key"))
    return serializer.dumps(email, salt="email-confirmation-salt")

def confirm_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY", "super-secret-dev-key"))
    try:
        email = serializer.loads(token, salt="email-confirmation-salt", max_age=expiration)
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

# --- SAVE / UNSAVE / GET SAVED ARTICLES ---
@aggregator_bp.route("/save-article", methods=["POST"])
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

@aggregator_bp.route("/unsave-article", methods=["POST"])
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

@aggregator_bp.route("/saved-articles")
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

# --- ZIP/LOCAL NEWS ENDPOINTS ---
def get_city_state_from_zip(zipcode):
    info = nomi.query_postal_code(zipcode)
    if pd.notna(info.place_name) and pd.notna(info.state_name):
        return info.place_name, info.state_name
    return None, None

async def fetch_google_local_feed(zipcode: str, limit: int = 50):
    city, state = get_city_state_from_zip(zipcode)
    if not city or not state:
        print(f"⚠️ Could not resolve ZIP {zipcode} to city/state")
        return []
    query = f"{city} {state} local news"
    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    print(f"📡 Fetching Google News RSS for: {query}")
    return await fetch_single_feed(url, limit=limit)

async def fetch_single_feed(url, limit=50):
    articles = []
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
            async with session.get(url, timeout=20) as response:
                raw_data = await response.read()
        feed = feedparser.parse(raw_data)
        cutoff = datetime.utcnow() - timedelta(days=7)
        for index, entry in enumerate(feed.entries[:limit]):
            title = entry.get("title", "No Title")
            desc = entry.get("summary", "")
            if not desc.strip():
                continue
            parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
            if not parsed_date:
                continue
            pub_date = datetime(*parsed_date[:6])
            if pub_date < cutoff:
                continue
            text = f"{title} {desc}"
            category = predict_category(text)
            if category == "Unknown":
                category = "General"
            summary = simple_summarize(desc)
            parsed_url = urlparse(url)
            source = parsed_url.netloc.replace("www.", "").replace("feeds.", "").split(".")[0].lower()
            article_id = generate_article_id(entry.get("link", f"{url}-{index}"))
            bias = detect_political_bias(f"{title}. {desc}", article_id=article_id, source=source)
            articles.append({
                "id": article_id,
                "title": title,
                "summary": summary,
                "description": desc,
                "url": entry.get("link", "#"),
                "category": category,
                "source": source,
                "published": pub_date.isoformat(),
                "bias": bias
            })
        articles.sort(key=lambda a: a["published"], reverse=True)
        print(f"✓ (async) {len(articles)} from {url}")
    except Exception as e:
        print(f"⚠️ (async) Failed to fetch {url}: {e}")
    return articles

@aggregator_bp.route("/news/local")
@login_required
def local_news_page():
    return render_template("local_news.html")

@aggregator_bp.route("/api/news/local")
@login_required
def get_local_news():
    zip_code = current_user.zipcode.strip()
    if not zip_code or not re.match(r"^\d{5}$", zip_code):
        return jsonify([])
    with local_cache_lock:
        articles = local_articles_cache.get(zip_code)
    if not articles:
        print(f"⚠️ No cached feed for ZIP {zip_code}, attempting live fetch...")
        articles = asyncio.run(fetch_google_local_feed(zip_code, limit=50))
        with local_cache_lock:
            local_articles_cache[zip_code] = articles
    print(f"✅ Returning {len(articles)} local articles to user {current_user.username}")
    return jsonify(articles)

@aggregator_bp.route("/update-zipcode", methods=["POST"])
@login_required
def update_zipcode():
    zip_input = request.form.get("zip") or (request.get_json() or {}).get("zip", "").strip()
    if zip_input and re.match(r"^\d{5}$", zip_input):
        current_user.zipcode = zip_input
        db.session.commit()
        flash("ZIP code updated successfully!", "success")
    else:
        flash("Invalid ZIP code format. Please enter a 5-digit U.S. ZIP.", "error")
    return redirect(url_for("aggregator.account_page"))

@aggregator_bp.route("/resend-confirmation", methods=["POST"])
@login_required
def resend_confirmation():
    if current_user.is_confirmed:
        return jsonify({"message": "Account already confirmed."}), 200
    token = generate_confirmation_token(current_user.email)
    send_confirmation_email(current_user.email, current_user.username, token)
    return jsonify({"message": "Confirmation email resent."}), 200

# --- BACKGROUND THREADS ---
def periodic_local_refresh_by_zip(interval=900):  # 15 min
    async def refresh_loop():
        while True:
            print("🔄 Refreshing local feeds from Google News RSS...")
            users = User.query.filter(User.zipcode.isnot(None)).all()
            seen_zips = set()
            tasks = []
            for user in users:
                zip_clean = user.zipcode.strip()
                if re.match(r"^\d{5}$", zip_clean) and zip_clean not in seen_zips:
                    seen_zips.add(zip_clean)
                    tasks.append(fetch_google_local_feed(zip_clean, limit=50))
            await asyncio.gather(*tasks)
            await asyncio.sleep(interval)
    threading.Thread(target=lambda: asyncio.run(refresh_loop()), daemon=True).start()

def start_background_tasks():
    threading.Thread(target=periodic_refresh, daemon=True).start()
    periodic_local_refresh_by_zip()

# --- End aggregator.py ---
# --- Background task starter ---
def start_background_tasks():
    threading.Thread(target=periodic_refresh, daemon=True).start()
    # Add other background tasks if needed

# --- End of aggregator.py ---
