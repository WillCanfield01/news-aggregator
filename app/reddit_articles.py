import os
import re
import openai
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify
from unidecode import unidecode
from app.models import CommunityArticle
from app import db
from datetime import date
from markdown2 import markdown
import praw
import markdown2
import requests

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
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

EXTRA_BANNED = [
    "hitler", "nazi", "terror", "rape", "porn", "sex", "nsfw",
    # ...add any other personal names, brands, or keywords you want to avoid in titles
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

def rewrite_title(original_title):
    # Remove banned words first
    text = unidecode(original_title)
    words = text.split()
    cleaned = [w for w in words if w.lower() not in BANNED_WORDS + EXTRA_BANNED]
    text = ' '.join(cleaned)
    
    # Now use OpenAI to rewrite for clarity/SEO
    prompt = (
        f"Rewrite this community discussion question as a clear, concise, and professional SEO article headline. "
        f"Remove any references to Reddit, NSFW, personal names, or internet slang. "
        f"Do not ask questions—make it a statement if possible. Example: "
        f"Input: 'What do you think was in the mystery box? Ghislaine'\n"
        f"Output: 'Unraveling the Mystery Box: Theories and Surprises'\n"
        f"Input: '{text}'\n"
        f"Output:"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.2,
    )
    headline = response.choices[0].message.content.strip()
    # Optionally strip trailing punctuation, excess spaces, or quotes
    headline = headline.strip(' "\'')
    return headline or "Untitled Article"

def save_article_db(title, content_md, filename, html_content=None):
    article = CommunityArticle(
        date=date.today(),
        filename=filename,
        title=title,
        content=content_md,
        html_content=html_content or markdown(content_md)
    )
    db.session.add(article)
    db.session.commit()
    return article.id

def get_unsplash_image(query):
    url = "https://api.unsplash.com/photos/random"
    params = {
        "query": query,
        "orientation": "landscape",
        "client_id": UNSPLASH_ACCESS_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        return data["urls"]["regular"], data["user"]["name"], data["links"]["html"]  # url, photographer, profile
    return None, None, None

def insert_image_markdown(md_text, image_url, alt_text, caption=None, after_heading=None):
    image_md = f"![{alt_text}]({image_url})"
    if caption:
        image_md += f"\n*{caption}*"
    lines = md_text.splitlines()
    if after_heading:
        # Insert after the heading that matches after_heading
        for i, line in enumerate(lines):
            if after_heading.lower() in line.lower():
                lines.insert(i+1, image_md)
                break
        else:
            lines.append(image_md)
    else:
        # Default: after first heading
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                lines.insert(i+1, image_md)
                break
        else:
            lines.insert(0, image_md)
    return "\n".join(lines)

import json

def suggest_image_sections_and_captions(article_md, outline):
    prompt = (
        "Given the following article outline and the full draft, suggest 1-3 sections where a relevant image would add value. "
        "For each section, provide:\n"
        "- The section heading or a short description of the content after which to insert the image\n"
        "- An image search query (e.g. 'family traditions', 'lost customs', 'community gathering')\n"
        "- A short, descriptive caption and alt text for the image\n\n"
        f"Outline:\n{outline}\n\nArticle Draft (Markdown):\n{article_md}\n\n"
        "Format your reply as JSON:\n"
        "[{\"section\": \"Section Heading\", \"query\": \"image search term\", \"caption\": \"caption and alt text\"}, ...]"
        "\nReturn only valid, parsable JSON, with no extra text, comments, or explanations."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    gpt_output = response.choices[0].message.content
    print("Raw image suggestion output:", gpt_output)
    try:
        # Parse the JSON output
        suggestions = json.loads(gpt_output)
        # If suggestions is a dict with a single key, extract the value
        if isinstance(suggestions, dict):
            suggestions = list(suggestions.values())[0]
        # If it's a string, try to parse again (sometimes double-encoded)
        if isinstance(suggestions, str):
            suggestions = json.loads(suggestions)
        if not isinstance(suggestions, list):
            suggestions = [suggestions]
        # Make sure all are dicts
        suggestions = [s for s in suggestions if isinstance(s, dict)]
        print("Parsed suggestions:", suggestions)
        return suggestions
    except Exception as e:
        print("AI JSON parse error:", e)
        return []

def extract_markdown_title(lines, fallback):
    # Find the first heading line, else fallback
    for line in lines:
        if line.strip().startswith("#"):
            return line.strip("# \n")
    return fallback

def generate_image_alt_text(headline, keywords, outline, section_text):
    prompt = (
        f"Write a highly descriptive, SEO-optimized alt text for an image to be used in an article with the headline: '{headline}'. "
        f"Target these keywords: {', '.join(keywords)}. "
        f"The image appears in this section: '{section_text}'. "
        "Alt text should be 8-20 words, clear, and relevant for both screen readers and Google image search. "
        "Do not use 'image of', 'photo of', etc."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=0.3,
    )
    alt_text = response.choices[0].message.content.strip().strip('"').strip("'")
    return alt_text

@bp.route("/")
def show_articles():
    articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
    return render_template("reddit_articles.html", articles=articles)

@bp.route("/generate", methods=["POST"])
def generate():
    fname = generate_article_for_today()
    return jsonify({"filename": fname, "success": True})

@bp.route("/articles")
def published_articles():
    articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
    for a in articles:
        # Use plain text from markdown, or truncate rendered HTML if you prefer
        import re
        # Remove markdown/image links and limit length
        plain = re.sub(r'\!\[.*?\]\(.*?\)', '', a.content)  # Remove images
        plain = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plain)  # [link text](url) → link text
        plain = re.sub(r'\*\*|\*|__|_', '', plain)  # Remove bold/italic
        # Get first 30 words or 300 chars as excerpt
        words = plain.split()
        a.excerpt = " ".join(words[:40]) + ("..." if len(words) > 40 else "")
    return render_template("published_articles.html", articles=articles)

@bp.route("/articles/<filename>")
def read_article(filename):
    article = CommunityArticle.query.filter_by(filename=filename).first_or_404()
    return render_template("single_article.html", content=article.html_content, title=article.title)
def clean_title(title):
    # Remove mentions of Reddit, AskReddit, r/AskReddit, etc.
    title = re.sub(r'\b([Rr]/)?AskReddit\b:?\s*', '', title)
    title = re.sub(r'\b[Rr]eddit\b:?\s*', '', title)
    return title.strip()

def generate_article_for_today():
    post = get_top_askreddit_post()
    cleaned_topic = clean_title(post["title"])
    headline = rewrite_title(cleaned_topic)
    keywords = extract_keywords(headline, post["comments"])
    outline = generate_outline(headline, keywords)
    article_md = generate_article(headline, outline, keywords)

    # --- IMAGE SUGGESTIONS & INSERTION HERE ---
    image_suggestions = suggest_image_sections_and_captions(article_md, outline)
    for suggestion in image_suggestions:
        image_url, photographer, image_page = get_unsplash_image(suggestion["query"])
        if image_url:
            article_md = insert_image_markdown(
                article_md, image_url,
                alt_text=suggestion["caption"],
                caption=f"{suggestion['caption']} (Photo by {photographer} on Unsplash)",
                after_heading=suggestion["section"]
            )
    # --- END IMAGE INSERTION ---
    html_content = markdown(article_md)
    filename = f"{datetime.now().strftime('%Y%m%d')}_{re.sub('[^a-zA-Z0-9]+', '-', headline)[:50]}"
    save_article_db(headline, article_md, filename, html_content)
    print(f"✅ Saved to DB: {filename}")
    return filename

if __name__ == "__main__":
    from app import create_app
    app = create_app()
    with app.app_context():
        print("Generating today's Reddit article...")
        fname = generate_article_for_today()
        print(f"✅ Generated and saved: {fname}")
