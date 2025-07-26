import os
import re
import requests
import openai
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify

import praw

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUBREDDIT = "AskReddit"

ARTICLES_DIR = os.path.join(os.path.dirname(__file__), "generated_articles")  # More portable

if not os.path.exists(ARTICLES_DIR):
    os.makedirs(ARTICLES_DIR)

openai.api_key = OPENAI_API_KEY

bp = Blueprint("reddit_articles", __name__, url_prefix="/reddit-articles")

def get_top_askreddit_post():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent="RealRoundup/1.0 by u/No-Plan-81"
    )
    subreddit = reddit.subreddit(SUBREDDIT)
    top = next(subreddit.top("day", limit=1))
    top.comments.replace_more(limit=0)
    top_comments = [c.body for c in top.comments[:5] if hasattr(c, "body")]
    return {
        "title": top.title,
        "selftext": top.selftext,
        "url": top.url,
        "comments": top_comments,
        "id": top.id,
    }

def extract_keywords(text, comments=[]):
    prompt = (
        f"Extract the top 10 keywords or phrases from the following Reddit post and its top replies:\n\n"
        f"Post: {text}\n\n"
        f"Top Comments:\n" + "\n".join(comments) +
        "\n\nList as comma-separated keywords only."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.2
    )
    kw_text = response["choices"][0]["message"]["content"]
    keywords = [kw.strip() for kw in re.split(r",|;", kw_text) if kw.strip()]
    return keywords

def generate_outline(topic, keywords):
    prompt = (
        f"Create a detailed SEO blog post outline for the topic '{topic}'. "
        f"Target these keywords: {', '.join(keywords)}. "
        "Include 5-7 headings/subheadings, meta title, meta description, and a FAQ section."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"]

def generate_article(topic, outline, keywords):
    prompt = (
        f"Using this outline:\n{outline}\n\n"
        f"Write a 1000+ word SEO blog article on '{topic}' targeting these keywords: {', '.join(keywords)}. "
        "Use clear sections, add H2/H3 headings, and make it fact-based. "
        "End with an FAQ. Avoid fluff, be engaging and original."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"]

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
    post = get_top_askreddit_post()
    keywords = extract_keywords(post["title"], post["comments"])
    outline = generate_outline(post["title"], keywords)
    article = generate_article(post["title"], outline, keywords)
    fname = save_article_md(post["title"], article)
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

def generate_article_for_today():
    post = get_top_askreddit_post()
    keywords = extract_keywords(post["title"], post["comments"])
    outline = generate_outline(post["title"], keywords)
    article = generate_article(post["title"], outline, keywords)
    fname = save_article_md(post["title"], article)
    print(f"Generated and saved: {fname}")
    return fname
