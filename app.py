from flask import Flask, jsonify
from flask_cors import CORS
import requests
from transformers import pipeline, AutoTokenizer, AutoModel
import os
import torch

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Fetch the API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"

# Load the distilbert-base-uncased model for encoding the text (use for extractive summarization)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to extract sentence embeddings and use them for extractive summarization
def summarize_text(text, max_length=100):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Getting sentence embedding by averaging hidden states

    # Here you could apply a more sophisticated method to rank or cluster sentences
    # For now, just returning the first 100 characters of the text as a placeholder summary
    return text[:max_length]

# Fetch and summarize news articles
@app.route("/news")
def get_news():
    try:
        response = requests.get(NEWS_URL)
        articles = response.json().get('articles', [])

        summarized_articles = []
        for article in articles:
            # Use distilbert-base-uncased model to summarize the article description
            if article['description']:  # Ensure there's a description available
                summary = summarize_text(article['description'], max_length=100)

                summarized_articles.append({
                    'title': article['title'],
                    'summary': summary,
                    'url': article['url']
                })

        return jsonify(summarized_articles)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)  # Use port 10000 for Render
  # Make sure this port matches your frontend setup
