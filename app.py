import os
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global classifier variable (lazy loading to prevent memory issues)
_classifier = None

# Categories for classification
CATEGORIES = ["Finance", "Sports", "Politics", "Entertainment", "Health", "Technology"]

# Fetch API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=100"

# Create a session for API requests (reduces overhead)
session = requests.Session()

def get_classifier():
    """ Lazily loads the classifier model to avoid memory issues. """
    global _classifier
    if _classifier is None:
        print("DEBUG: Initializing Hugging Face model...")
        try:
            _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            print("DEBUG: Model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load model - {e}")
            _classifier = None
    return _classifier

def predict_category(article_text):
    """ Predicts the category of an article using the Hugging Face model. """
    classifier = get_classifier()
    if classifier is None:
        print("ERROR: Model not loaded. Returning 'Unknown'.")
        return "Unknown"
    
    try:
        result = classifier(article_text, candidate_labels=CATEGORIES)
        return result["labels"][0]
    except Exception as e:
        print(f"ERROR: Model inference failed - {e}")
        return "Unknown"

def simple_summarize(text, max_words=50):
    """ Simple text summarization by truncating to max_words. """
    if not text:
        return "No description available."
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def fetch_news():
    """ Fetches news articles, classifies them, and returns structured data. """
    try:
        print("DEBUG: Fetching news articles...")
        response = session.get(NEWS_URL, timeout=5)
        response.raise_for_status()
        
        articles = response.json().get("articles", [])
        print(f"DEBUG: {len(articles)} articles fetched.")
        
        categorized_articles = []
        for article in articles:
            title = article.get("title", "No Title")
            description = article.get("description", "")
            article_text = title + " " + description
            category = predict_category(article_text)
            
            categorized_articles.append({
                "title": title,
                "summary": simple_summarize(description, max_words=50),
                "url": article.get("url"),
                "category": category
            })
        
        return categorized_articles
    
    except requests.RequestException as e:
        print(f"ERROR: News API request failed - {e}")
        return []
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}")
        return []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news")
def get_news():
    articles = fetch_news()
    print(f"DEBUG: Returning {len(articles)} articles.")
    return jsonify(articles)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)