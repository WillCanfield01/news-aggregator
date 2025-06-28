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
from html import unescape
from flask import Flask, jsonify, render_template, request, session
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

MAX_CACHED_ARTICLES = 300
os.environ["TOKENIZERS_PARALLELISM"] = "false"
postmark = PostmarkClient(server_token=os.getenv("POSTMARK_SERVER_TOKEN"))

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

CORS(app, supports_credentials=True)
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "home"

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(200))
    saved_articles = db.relationship("SavedArticle", backref="user", lazy=True)

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
        login_user(user)
        return jsonify(success=True, username=user.username)
    else:
        return jsonify(success=False, message="Invalid credentials"), 401

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
    return render_template("account.html", username=current_user.username, saved_articles=saved)

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

    if not username or not password or not email:
        return jsonify({"error": "All fields are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    try:
        user = User(username=username, email=email)
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
        return redirect(url_for('home'))  # Or use render_template("email_confirmed.html")

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


if __name__ == "__main__":
    preload_articles_batched(RSS_FEED_BATCHES[0], use_ai=False)  # 🧠 Preload once immediately
    threading.Thread(target=periodic_refresh, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))