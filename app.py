import os
from flask import Flask, jsonify
from flask_cors import CORS
import requests
from transformers import pipeline
import torch

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Fetch the API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

# Load the summarizer model from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/")
def home():
    return "News Aggregator API is running!"

@app.route("/news")
def get_news():
    try:
        response = requests.get(NEWS_URL)
        articles = response.json().get('articles', [])

        summarized_articles = []
        for article in articles:
            if article['description']:
                input_length = len(article['description'].split())
                max_length = min(100, input_length + 20)

                with torch.no_grad():
                    summary = summarizer(article['description'], max_length=max_length, min_length=50, do_sample=False)

                summarized_articles.append({
                    'title': article['title'],
                    'summary': summary[0]['summary_text'],
                    'url': article['url']
                })

        return jsonify(summarized_articles)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use the PORT variable from Render
    app.run(host="0.0.0.0", port=port)

