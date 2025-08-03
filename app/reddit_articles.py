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
import difflib
import re
import random

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
SUBREDDIT = "AskReddit"

ARTICLES_DIR = os.path.join(os.path.dirname(__file__), "generated_articles")
if not os.path.exists(ARTICLES_DIR):
    os.makedirs(ARTICLES_DIR)

openai.api_key = OPENAI_API_KEY

bp = Blueprint("all-articles", __name__, url_prefix="/all-articles")

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

GPTISMS = [
    "delve", "dive into", "navigate", "vibrant", "comprehensive", "pivotal", "notably",
    "realm", "landscape", "tapestry", "embark", "unpack", "delving", "navigating", "explore", 
    "in today's world", "in today's society", "at the end of the day", "from all walks of life",
    "it's worth noting", "in conclusion", "ultimately", "delving", "navigating", "pivot", "essentially", "moreover",
]

def sanitize_gptisms(text):
    for phrase in GPTISMS:
        # Remove exact matches or common phrasal forms
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text, flags=re.IGNORECASE)
    # Remove double spaces created by removal
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def render_faq(qa_pairs):
    result = []
    for idx, qa in enumerate(qa_pairs, 1):
        q = qa['q']
        a = qa['a']
        # Add extra line breaks for clarity
        result.append(f"**Q{idx}:** {q}\n\n**A:** {a}\n")
    return "\n".join(result)

def humanize_reflection(text):
    quirks = [
        lambda s: re.sub(r"\bthe\b", "teh", s, count=1),
        lambda s: re.sub(r"\b(I think|I guess|maybe|honestly)\b", r"\1, \1", s, count=1),
        lambda s: s.rstrip(".!?") + " I dunno.",  # strip punctuation
        lambda s: s.rstrip(".!?") + " Just saying.",
        lambda s: re.sub(r"\.$", "…", s, 1),
        lambda s: s[:random.randint(int(len(s)*0.6), int(len(s)*0.85))].rstrip(".!?") + "…"  # abrupt cut
    ]
    if random.random() < 0.40:
        quirk = random.choice(quirks)
        text = quirk(text)
    return text

def is_safe(text):
    text = text.lower()
    for word in BANNED_WORDS:
        if word in text:
            return False
    return True

def split_markdown_sections(md):
    # Only split on markdown headings (## or #), not numbered lines
    pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(md))
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(md)
        heading = match.group(2).strip()
        content = md[start:end].strip()
        sections.append((heading, content))
    return sections

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
        "Include 5-7 headings/subheadings, a meta title, meta description, and an FAQ section.\n"
        "Format your response as:\n"
        "Meta Title: ...\nMeta Description: ...\nOutline:\n# ... (markdown headings, etc)"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350,
        temperature=0.3,
    )
    content = response.choices[0].message.content
    mt = re.search(r"Meta Title:\s*(.+)", content)
    md = re.search(r"Meta Description:\s*(.+)", content)
    ol = re.search(r"Outline:\s*([\s\S]+)", content)
    meta_title = mt.group(1).strip() if mt else topic
    meta_description = md.group(1).strip() if md else ""
    outline = ol.group(1).strip() if ol else content
    return meta_title, meta_description, outline

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

def fix_broken_personal_note(note):
    # Collapse 2+ newlines to a single space
    note = re.sub(r'\n{2,}', ' ', note)
    # Remove isolated italicized words/lines (e.g. "*really*")
    note = re.sub(r'\n?\*(\w+)\*\n?', '', note)
    # Remove lingering single-word lines
    note = re.sub(r'\n([a-zA-Z]{1,10})\n', ' ', note)
    # Collapse awkward linebreaks inside sentences
    note = re.sub(r'\n+', ' ', note)
    # Remove leading/trailing whitespace
    note = note.strip()
    # Optionally: Truncate to 2 sentences if too long
    sentences = re.split(r'(?<=[.!?]) +', note)
    if len(sentences) > 2:
        note = ' '.join(sentences[:2])
    return note

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

def save_article_db(title, content_md, filename, html_content=None, meta_title=None, meta_description=None):
    article = CommunityArticle(
        date=date.today(),
        filename=filename,
        title=title,
        content=content_md,
        html_content=html_content or markdown(content_md),
        meta_title=meta_title,
        meta_description=meta_description
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
        # Pull Unsplash's own alt description if present
        alt = data.get("alt_description") or data.get("description") or ""
        return data["urls"]["regular"], data["user"]["name"], data["links"]["html"], alt
    return None, None, None, None

def insert_image_markdown(md_text, image_url, alt_text, caption=None, after_heading=None):
    image_md = f"![{alt_text}]({image_url})"
    if caption:
        image_md += f"\n*{caption}*"
    lines = md_text.splitlines()
    inserted = False
    if after_heading:
        headings = [line for line in lines if line.strip().startswith("#")]
        close = difflib.get_close_matches(after_heading, headings, n=1, cutoff=0.5)
        if close:
            idx = lines.index(close[0])
            lines.insert(idx + 1, image_md)
            inserted = True
        else:
            lines.append(image_md)
            inserted = True
    else:
        # Fallback: insert after first heading
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                lines.insert(i + 1, image_md)
                inserted = True
                break
        if not inserted:
            lines.insert(0, image_md)
    return "\n".join(lines)

import json

def generate_section_image_suggestion(headline, keywords, outline, section_heading, section_content):
    prompt = (
        f"Given the article '{headline}' (keywords: {', '.join(keywords)}), "
        f"and the section below:\n\n"
        f"Section Heading: {section_heading}\n"
        f"Section Content:\n{section_content}\n\n"
        "Suggest a relevant image for Unsplash, with:\n"
        "- An image search query (max 5 words)\n"
        "- A short, descriptive caption and alt text (max 20 words)\n"
        "Format as JSON: {\"query\": \"...\", \"caption\": \"...\"}\n"
        "If no image fits, reply: {\"query\": \"\", \"caption\": \"skip\"}"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    try:
        suggestion = json.loads(response.choices[0].message.content)
        # Only use if not "skip"
        if suggestion.get("caption", "").lower() == "skip" or not suggestion.get("query"):
            return None
        suggestion["section"] = section_heading
        return suggestion
    except Exception as e:
        print("Section image parse error:", e)
        return None

def suggest_image_sections_and_captions(article_md, outline):
    prompt = (
        "Given the article outline and draft below, suggest **EXACTLY 4 distinct sections** where a relevant image would add value. "
        "For EACH section, provide:\n"
        "- The full section heading (verbatim, exactly as in the draft)\n"
        "- An image search query for Unsplash\n"
        "- A descriptive caption (also used as alt text)\n\n"
        "DO NOT return fewer or more than 4 items. Format your reply as a valid JSON list, e.g.:\n"
        "[\n"
        "  {\"section\": \"1. Introduction to X\", \"query\": \"conceptual intro image\", \"caption\": \"A conceptual image about X...\"},\n"
        "  {\"section\": \"2. How Y Works\", \"query\": \"technical process\", \"caption\": \"Diagram of how Y works...\"},\n"
        "  {\"section\": \"3. Benefits of Z\", \"query\": \"happy team working\", \"caption\": \"Happy team collaborating at work...\"},\n"
        "  {\"section\": \"4. Common Mistakes\", \"query\": \"warning sign business\", \"caption\": \"Warning sign in a business setting...\"}\n"
        "]\n"
        "Return only valid, parsable JSON and always exactly 4 items—never more, never less.\n\n"
        f"Outline:\n{outline}\n\nArticle Draft:\n{article_md}\n"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=340,
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    gpt_output = response.choices[0].message.content
    print("Raw image suggestion output:", gpt_output)
    try:
        suggestions = json.loads(gpt_output)
        # If GPT returns {"result": [...]}, use that list
        if isinstance(suggestions, dict) and "result" in suggestions and isinstance(suggestions["result"], list):
            suggestions = suggestions["result"]
        elif isinstance(suggestions, dict):
            suggestions = [suggestions]
        elif not isinstance(suggestions, list) or suggestions is None:
            suggestions = []
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
    # Remove the first markdown heading (e.g. "# Title") before rendering
    def remove_first_heading(md_text):
        lines = md_text.splitlines()
        found = False
        output = []
        for line in lines:
            if not found and line.strip().startswith("#"):
                found = True
                continue  # skip this heading line
            output.append(line)
        return "\n".join(output)
    cleaned_md = remove_first_heading(article.content)
    html_content = markdown(cleaned_md)
    
    # Add meta_title and meta_description support
    meta_title = None
    meta_description = None
    # If you have fields for these, use them
    if hasattr(article, "meta_title") and article.meta_title:
        meta_title = article.meta_title
    if hasattr(article, "meta_description") and article.meta_description:
        meta_description = article.meta_description
    else:
        # Fallback: Use first ~160 chars of plain text
        plain = re.sub(r'<.*?>', '', html_content)  # Remove HTML tags
        meta_description = plain[:160]

    return render_template(
        "single_article.html",
        title=article.title,
        date=article.date,
        content=html_content,
        meta_title=meta_title,
        meta_description=meta_description
    )

def remove_gpt_dashes(text):
    # Replace most GPT em dashes with commas, except in number ranges
    # 1. Spaces around — become comma
    text = re.sub(r'(\s)—(\s)', r'\1,\2', text)
    # 2. Standalone em dash after word, not before a number, becomes comma
    text = re.sub(r'([a-zA-Z]),?\s*—\s*([a-zA-Z])', r'\1, \2', text)
    # 3. If double em dashes (rare), replace with single
    text = re.sub(r'—{2,}', '—', text)
    return text

def strip_unwanted_bold(text):
    # Only keep bold around "Personal Note:"
    # 1. Remove all ** that don't wrap "Personal Note:"
    # 2. Optionally, remove single * too if you see stray italics
    # (But keep bold for section headings if needed)
    # Remove ** unless at "> **Personal Note:**"
    def replacer(match):
        # Only keep if it's a proper Personal Note label
        content = match.group(1)
        if content.strip().lower() == 'personal note:':
            return f"**Personal Note:**"
        else:
            return content

    # This replaces any **...** except the label
    text = re.sub(r"\*\*(.*?)\*\*", replacer, text)

    # Optionally, remove single * (italics) everywhere
    # text = re.sub(r"\*(.*?)\*", r"\1", text)

    return text

def clean_title(title):
    # Remove mentions of Reddit, AskReddit, r/AskReddit, etc.
    title = re.sub(r'\b([Rr]/)?AskReddit\b:?\s*', '', title)
    title = re.sub(r'\b[Rr]eddit\b:?\s*', '', title)
    return title.strip()

MAX_IMAGE_ATTEMPTS = 2  # Or 3 if you want

def get_image_suggestions(article_md, outline, min_images=3, max_images=5):
    for attempt in range(MAX_IMAGE_ATTEMPTS):
        suggestions = suggest_image_sections_and_captions(article_md, outline)
        # If dict, wrap in a list!
        if isinstance(suggestions, dict):
            suggestions = [suggestions]
        if isinstance(suggestions, list) and len(suggestions) >= min_images:
            return suggestions[:max_images]
        print(f"⚠️ Attempt {attempt+1}: Only {len(suggestions)} suggestions. Retrying...")
    print("⚠️ Could not get enough image suggestions after retrying.")
    return suggestions if isinstance(suggestions, list) else [suggestions]

def generate_personal_intro(topic):
    prompt = (
        f"Write a short, authentic introduction (2-4 sentences) for an article on '{topic}'. "
        "Make it sound like a real person: share a personal memory, opinion, or curiosity about the topic, using a warm, conversational tone."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()

def generate_personal_reflection(topic, section_heading, section_content):
    prompt = (
        f"As the author, pause and add a personal aside in first-person. "
        f"Let yourself react to what you just wrote about '{section_heading}'—share a memory, an honest opinion, a frustration, a curiosity, or even a confession. "
        f"Vary the opener and dont always use phrases like 'Honestly' or 'I wonder.' Sometimes use humor, sometimes curiosity, sometimes mild skepticism, and sometimes just a single, authentic sentence."
        f"Make sure the format of personl notes are always varied. DO NOT SOUND LIKE A ROBOT. We want the personal notes to all be unique and human not formulatic."
        f"Write as if you're journaling or talking to a friend, not as a separate reader. "
        f"Do NOT start with 'Reading this section' or 'Reading about.' Avoid restating what was already said. Instead, make it feel authentic, like your own voice coming through. "
        f"Relate to it honestly: share a feeling, a memory, a frustration, or a personal opinion. "
        f"Imagine you're telling a friend what this section made you think about, or how it connects to your life or curiosity. "
        f"Don't just summarize. If you don't have a strong opinion, share a small doubt, hope, or question. "
        "Each reflection must sound totally distinct—use different tones, openers, and styles every time. "
        "Avoid starting more than one note in the same way. Basic examples are: 'You know,' 'Sometimes I wonder,' or 'It still surprises me'. We want change it up every time and make them more human and varied then this."
        "No cliches or generic intros. Avoid phrases like 'I catch myself,' 'sometimes I wonder,' or 'it makes me think.' Get specific, be quirky, or say something a little odd. "
        "If you can't think of anything, write a mini-confession or random opinion."
        "Keep it first-person, but vary your level of confidence and formality. Embrace oddball takes and variety!\n\n"
        f"\n\nSection Heading: {section_heading}\n"
        f"Section Content: {section_content[:400]}\n"
        "Length: 1-2 sentences. Write like a real, thoughtful person, not a robot."
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=56,
        temperature=1.2,
    )
    return sanitize_gptisms(response.choices[0].message.content.strip())

def parse_faq_section(faq_text):
    # Split into Q/A pairs using regex
    qa_pairs = []
    # Regex: match Q (or Question), then everything until next Q or end
    pattern = re.compile(
        r'(?:Q(?:uestion)?[:\s]+)(.+?)(?:\nA(?:nswer)?[:\s]+(.+?))?(?=\nQ|$)', 
        re.DOTALL | re.IGNORECASE
    )
    for match in pattern.finditer(faq_text):
        question = match.group(1).strip()
        answer = (match.group(2) or '').strip()
        if question:
            qa_pairs.append({'q': question, 'a': answer})
    return qa_pairs

def humanize_faq_answer(answer):
    openers = [
        "Honestly, I'm not totally sure. ",
        "Depends who you ask. ",
        "I guess it depends, right? ",
        "No idea, to be honest. ",
        ""
    ]
    # ~20% of time, prepend a "human" uncertainty
    if random.random() < 0.20:
        answer = random.choice(openers) + answer
    answer = sanitize_gptisms(answer)
    answer = humanize_reflection(answer)
    return answer

def generate_lighthearted_aside(topic, section_heading=None):
    if section_heading:
        prompt = (
            f"Write a one-sentence lighthearted joke, witty observation, or relatable aside about the section '{section_heading}' "
            f"in an article about '{topic}'. Make it friendly, not forced—think of something a clever friend would say in passing."
        )
    else:
        prompt = (
            f"Write a short (1 sentence) playful joke, pun, or funny thought related to the topic '{topic}'. "
            "Keep it natural, like something you'd say to make a reader smile. Avoid anything cringey or over-the-top."
        )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()

def assemble_article_with_humor(headline, sections):
    new_sections = []
    humor_count = 0
    max_humor = 2  # Don’t overdo it

    for i, (heading, content) in enumerate(sections):
        # Always add the section
        new_sections.append(f"{heading}\n{content}")

        # Don't add humor after the last section
        if i < len(sections) - 1:
            # 30% chance to add humor, max 2 per article
            if random.random() < 0.3 and humor_count < max_humor:
                aside = generate_lighthearted_aside(headline, heading)
                new_sections.append(f"\n😅 {aside}\n")
                humor_count += 1

    return "\n\n".join(new_sections)

def generate_article_for_today():
    post = get_top_askreddit_post()
    cleaned_topic = clean_title(post["title"])
    headline = rewrite_title(cleaned_topic)
    keywords = extract_keywords(headline, post["comments"])
    meta_title, meta_description, outline = generate_outline(headline, keywords)
    article_md = generate_article(headline, outline, keywords)

    personal_intro = generate_personal_intro(headline)
    article_with_human = personal_intro.strip() + "\n\n"

    sections = split_markdown_sections(article_md)
    img_count = 0
    faq_section = None

    for i, (heading, content) in enumerate(sections):
        heading_lower = heading.lower()
        is_faq = heading_lower.startswith("faq")
        is_conclusion = heading_lower.startswith("conclusion")

        # Show heading, except for FAQ or Conclusion which get their own style
        if not (is_faq or is_conclusion):
            article_with_human += f"## {heading}\n\n"
        elif is_conclusion:
            article_with_human += f"## Conclusion\n\n"
        elif is_faq:
            faq_section = (heading, content)
            continue

        # Add image after heading if appropriate
        if (
            img_count < 5 and content and len(content.strip()) > 30
            and not is_faq and not is_conclusion
        ):
            suggestion = generate_section_image_suggestion(
                headline, keywords, outline, heading, content
            )
            if suggestion:
                image_url, photographer, image_page, unsplash_alt = get_unsplash_image(suggestion.get("query", ""))
                caption = unsplash_alt or suggestion.get("caption", "Stock photo")
                if image_url:
                    article_with_human += f"![{caption}]({image_url})\n*Photo by {photographer} on Unsplash*\n\n"
                    img_count += 1

        # Strip repeated heading line from start of content (if OpenAI echos it inside)
        content_lines = content.splitlines()
        if content_lines:
            # Look for heading at start of content (case-insensitive, strip non-alphanum)
            content_first = re.sub(r'\W+', '', content_lines[0].lower())
            heading_cmp = re.sub(r'\W+', '', heading.lower())
            if heading_cmp and heading_cmp in content_first:
                content_lines = content_lines[1:]
        body = "\n".join(content_lines).strip()
        article_with_human += body + "\n\n"

        if not (is_faq or is_conclusion):
            reflection = generate_personal_reflection(headline, heading, body)
            reflection = fix_broken_personal_note(reflection)
            article_with_human += f"> **Personal Note:** {reflection}\n\n"

        if (i < len(sections) - 1) and not is_faq and not is_conclusion:
            article_with_human += "---\n"

    # Insert FAQ section, cleaned up
    if faq_section:
        faq_lines = faq_section[1].strip()
        qa_pairs = parse_faq_section(faq_lines)
        if qa_pairs:  # Only render if there's content!
            article_with_human += "\n## FAQ\n\n"
            for idx, qa in enumerate(qa_pairs, 1):
                question = qa['q']
                answer = humanize_faq_answer(qa['a'])
                article_with_human += f"**Q{idx}: {question}**\n\n{answer}\n\n"

    article_with_human = remove_gpt_dashes(article_with_human)
    article_with_human = strip_unwanted_bold(article_with_human)

    html_content = markdown(article_with_human)
    filename = f"{datetime.now().strftime('%Y%m%d')}_{re.sub('[^a-zA-Z0-9]+', '-', headline)[:50]}"
    if CommunityArticle.query.filter_by(filename=filename).first():
        print(f"⚠️ Article for filename {filename} already exists. Skipping save.")
        return filename
    save_article_db(headline, article_with_human, filename, html_content, meta_title, meta_description)
    print(f"✅ Saved to DB: {filename}")
    return filename

if __name__ == "__main__":
    from app import create_app
    app = create_app()
    with app.app_context():
        print("Generating today's Reddit article...")
        fname = generate_article_for_today()
        print(f"✅ Generated and saved: {fname}")
