import os
import re
import openai
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify

import praw

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUBREDDIT = "AskReddit"

ARTICLES_DIR = os.path.join(os.path.dirname(__file__), "generated_articles")
if not os.path.exists(ARTICLES_DIR):
    os.makedirs(ARTICLES_DIR)

openai.api_key = OPENAI_API_KEY

bp = Blueprint("reddit_articles", __name__, url_prefix="/reddit-articles")

BANNED_WORDS = [
    "sex", "sexual", "nsfw", "porn", "nude", "nudes", "vagina", "penis", "erection",
    "boobs", "boob", "breast", "cum", "orgasm", "masturbat", "anal", "ass", "butt",
    "dick", "cock", "blowjob", "suck", "f***", "shit", "piss", "rape", "molest",
    "incest", "adult", "fetish", "taboo", "explicit", "onlyfans"
    # ...add more as needed
]

def is_safe(text):
    text = text.lower()
    for word in BANNED_WORDS:
        if word in text:
            return False
    return True

def get_top_askreddit_post():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="RealRoundup/1.0 by u/No-Plan-81"
    )
    subreddit = reddit.subreddit(SUBREDDIT)
    # Get top 10 posts of the day, check each
    for post in subreddit.top(time_filter="day", limit=10):
        if getattr(post, "over_18", False):
            continue  # Skip if Reddit marks as NSFW
        if not is_safe(post.title) or not is_safe(post.selftext or ""):
            continue
        post.comments.replace_more(limit=0)
        # Only keep safe comments
        safe_comments = [c.body for c in post.comments[:10] if hasattr(c, "body") and is_safe(c.body)]
        if safe_comments:
            return {
                "title": post.title,
                "selftext": post.selftext,
                "url": post.url,
                "comments": safe_comments[:5],  # top 5 safe comments
                "id": post.id,
            }
    raise Exception("No safe AskReddit post found today!")

def extract_keywords(text, comments=[]):
    prompt = (
        f"Extract the top 10 keywords or phrases from the following user conversation and its most insightful replies:\n\n"
        f"Main Question: {text}\n\n"
        f"Top Replies:\n" + "\n".join(comments) +
        "\n\nList as comma-separated keywords only."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.2,
    )
    kw_text = response.choices[0].message.content
    keywords = [kw.strip() for kw in re.split(r",|;", kw_text) if kw.strip()]
    return keywords


def generate_outline(topic, keywords):
    prompt = (
        f"Create a detailed SEO blog post outline for the topic '{topic}'. "
        f"Target these keywords: {', '.join(keywords)}. "
        "The article should read like a trending community discussion, as if curated for a smart, independent advice site. "
        "Absolutely avoid any mention of Reddit, forums, or social media. "
        "Use a conversational style, sharing personal insights and tips. "
        "Include 5-7 headings/subheadings, a meta title, meta description, and an FAQ section."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_article(topic, outline, keywords):
    prompt = (
        f"Using this outline:\n{outline}\n\n"
        f"Write a 1000+ word SEO blog article on '{topic}' targeting these keywords: {', '.join(keywords)}. "
        "Write it as a helpful, original, and engaging advice column—share insights and practical wisdom, as if from a personal blog or expert contributor. "
        "Absolutely avoid any mention of Reddit, forums, or social media. The article must be fully independent. End with an FAQ."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.5,
    )
    return response.choices[0].message.content


def save_article_md(title, content):
    filename = f"{ARTICLES_DIR}/{datetime.now().strftime('%Y%m%d')}_{re.sub('[^a-zA-Z0-9]+', '-', title)[:50]}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(content)
    return filename


@bp.route("/")
def show_articles():
    files = [f for f in os.listdir(ARTICLES_DIR) if f.endswith(".md")]
    articles = []
    for fname in sorted(files, reverse=True):
        with open(os.path.join(ARTICLES_DIR, fname), encoding="utf-8") as f:
            content = f.read()
        articles.append({"filename": fname, "content": content[:500] + "...", "title": fname})
    return render_template("reddit_articles.html", articles=articles)


@bp.route("/generate", methods=["POST"])
def generate():
    fname = generate_article_for_today()
    return jsonify({"filename": fname, "success": True})


@bp.route("/articles")
def published_articles():
    files = [f for f in os.listdir(ARTICLES_DIR) if f.endswith(".md")]
    articles = []
    for fname in sorted(files, reverse=True):
        with open(os.path.join(ARTICLES_DIR, fname), encoding="utf-8") as f:
            content = f.read()
        title = fname.replace(".md", "")
        articles.append({
            "filename": fname,
            "title": title,
            "content": content[:400] + "...",  # Snippet/preview only
        })
    return render_template("published_articles.html", articles=articles)


@bp.route("/articles/<filename>")
def read_article(filename):
    path = os.path.join(ARTICLES_DIR, filename)
    if not os.path.exists(path):
        return "Article not found.", 404
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return render_template("single_article.html", content=content, title=filename.replace(".md", ""))

def clean_title(title):
    # Remove mentions of Reddit, AskReddit, r/AskReddit, etc.
    title = re.sub(r'\b([Rr]/)?AskReddit\b:?\s*', '', title)
    title = re.sub(r'\b[Rr]eddit\b:?\s*', '', title)
    return title.strip()

def generate_article_for_today():
    post = get_top_askreddit_post()
    clean_topic = clean_title(post["title"])
    keywords = extract_keywords(clean_topic, post["comments"])
    outline = generate_outline(clean_topic, keywords)
    article = generate_article(clean_topic, outline, keywords)
    fname = save_article_md(clean_topic, article)
    print(f"Generated and saved: {fname}")
    return fname

if __name__ == "__main__":
    # This lets you run: python app/reddit_articles.py
    print("Generating today's Reddit article...")
    fname = generate_article_for_today()
    print(f"✅ Generated and saved: {fname}")
