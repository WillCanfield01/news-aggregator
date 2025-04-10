import os
import requests
import torch
import time
import torch.nn.functional as F
import feedparser
import re
from html import unescape
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import openai

client = openai.OpenAI(api_key="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")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# RSS feed sources
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

# Preload model at startup for performance
print("DEBUG: Preloading model...")
_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/tweet-topic-21-multi"
).to("cpu").eval()
config = AutoConfig.from_pretrained("cardiffnlp/tweet-topic-21-multi")
_model_labels = list(config.id2label.values())
print("DEBUG: Model and labels preloaded:", _model_labels)

# Categories used for filtering
FILTERED_CATEGORIES = set([
    "Arts_&_culture", "Business_&_entrepreneurs", "Celebrity_&_pop_culture",
    "Diaries_&_daily_life", "Family", "Fashion_&_style", "Film_tv_&_video",
    "Fitness_&_health", "Food_&_dining", "Gaming", "Learning_&_educational",
    "Music", "News_&_social_concern", "Other_hobbies", "Relationships",
    "Science_&_technology", "Sports", "Travel_&_adventure", "Youth_&_student_life",
    "Entertainment", "Health", "Politics", "Finance", "Technology", "Unknown"
])

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

def normalize_category(category):
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
    return category_map.get(category.lower().replace(" ", "_"), category.title())

def predict_category(article_text, confidence_threshold=0.5):
    if not article_text.strip():
        return "Unknown"

    inputs = _tokenizer(article_text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad():
        outputs = _model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_index].item()

    if predicted_index >= len(_model_labels):
        return "Unknown"

    raw_label = _model_labels[predicted_index].lower().replace(" ", "_")
    normalized_category = normalize_category(raw_label)

    if confidence >= confidence_threshold:
        return normalized_category

    return "Unknown"

def strip_html(text):
    # Remove HTML tags and decode HTML entities
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean)

def simple_summarize(text, max_words=50):
    if not text:
        return "No description available."
    clean_text = strip_html(text)
    words = clean_text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# Corrected OpenAI Summarization Function with Timeout
def summarize_with_openai(article_text):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",  # Use the latest model you're working with
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize the following article, and make sure the summary is easy to understand, also make sure the results sound human and remain politically neutral: {article_text}"}
            ],
            max_tokens=75,
            temperature=0.5,
            timeout=30  # Set timeout for the request
        )

        # Access the summary correctly using the updated API response structure
        response_message = completion.choices[0].message.content.strip()
        return response_message

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to summarize article."

def fetch_news_from_rss():
    print("DEBUG: Fetching news from RSS feeds...")
    all_articles = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            print(f"DEBUG: {len(feed.entries)} entries from {url}")

            for entry in feed.entries[:10]:
                title = entry.get("title", "No Title")
                description = entry.get("summary", "")
                link = entry.get("link", "#")
                text = f"{title} {description}"

                # Ensure that the description is not empty
                if description.strip():
                    # Predict category
                    category = predict_category(text)

                    # If the category is in the allowed list, add it to the article list
                    if category in FILTERED_CATEGORIES:
                        # Get AI Summary from OpenAI API
                        ai_summary = summarize_with_openai(description)
                        all_articles.append({
                            "title": title,
                            "summary": ai_summary,  # Use the AI-generated summary
                            "url": link,
                            "category": category
                        })

        except Exception as e:
            print(f"ERROR parsing RSS feed {url}: {e}")

    print(f"DEBUG: Total articles from RSS: {len(all_articles)}")
    return all_articles

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news")
def get_news():
    articles = fetch_news_from_rss()
    return jsonify(articles)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

