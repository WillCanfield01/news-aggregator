import os
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load a pre-trained Hugging Face model for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the possible categories for classification
categories = ["Finance", "Sports", "Politics", "Entertainment", "Health", "Technology"]

# Fetch API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=100"

# Create a session for API requests (reduces overhead)
session = requests.Session()

# Simple summarization function
def simple_summarize(text, max_words=50):
    if not text:
        return "No description available."
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# Function to predict category using Hugging Face model
def predict_category(article_text):
    if not article_text:
        return "Unknown"
    result = classifier(article_text, candidate_labels=categories)
    return result["labels"][0]

# Function to fetch news articles
def fetch_news():
    try:
        print("DEBUG: Fetching news articles...")
        response = session.get(NEWS_URL, timeout=5)
        response.raise_for_status()
        
        articles = response.json().get("articles", [])
        print(f"DEBUG: {len(articles)} articles fetched.")
        
        categorized_articles = []
        for article in articles:
            title = article.get("title")
            description = article.get("description", "")

            if title:
                article_text = title + " " + (description if description else "")
                category = predict_category(article_text)
                categorized_articles.append({
                    "title": title,
                    "summary": simple_summarize(description, max_words=50),
                    "url": article.get("url"),
                    "category": category
                })
        
        return categorized_articles
    
    except requests.RequestException as e:
        print(f"News API request failed: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
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