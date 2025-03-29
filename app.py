import os
from flask import Flask, jsonify
from flask_cors import CORS
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Fetch the API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

# Load the summarizer model from Hugging Face explicitly (load only once at the start)
model_name = "distilbert-base-uncased"
summarizer = pipeline("summarization", model=AutoModelForSequenceClassification.from_pretrained(model_name), tokenizer=AutoTokenizer.from_pretrained(model_name))

# Fetch and summarize news articles
import torch

# Fetch and summarize news articles
@app.route("/news")
def get_news():
    try:
        response = requests.get(NEWS_URL)
        articles = response.json().get('articles', [])

        summarized_articles = []
        for article in articles:
            if article['description']:  # Ensure there's a description available
                # Use torch.no_grad() to save memory during inference
                with torch.no_grad():
                    summary = summarizer(article['description'], max_length=100, min_length=50, do_sample=False)

                summarized_articles.append({
                    'title': article['title'],
                    'summary': summary[0]['summary_text'],
                    'url': article['url']
                })

        return jsonify(summarized_articles)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Ensure it uses the PORT environment variable from Render
    app.run(host='0.0.0.0', port=port, debug=True)
