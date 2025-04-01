import os
import requests
from flask import Flask, jsonify, render_template
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Fetch API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

# Create a session for API requests (reduces overhead)
session = requests.Session()

# Simple summarization function (instead of Hugging Face)
def simple_summarize(text, max_words=50):
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/news")
def get_news():
    try:
        response = session.get(NEWS_URL, timeout=5)
        response.raise_for_status()  # Raise error for HTTP issues

        articles = response.json().get("articles", [])
        summarized_articles = [
            {
                "title": article["title"],
                "summary": simple_summarize(article["description"], max_words=50),
                "url": article["url"]
            }
            for article in articles if article.get("description")
        ]

        return jsonify(summarized_articles)

    except requests.RequestException as e:
        return jsonify({"error": f"News API request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
