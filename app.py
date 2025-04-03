import os
import requests
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Fetch API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=100"

# Create a session for API requests (reduces overhead)
session = requests.Session()

# Lazy-load model variables
_tokenizer = None
_model = None
_model_labels = None  # Stores actual category labels from the model

def load_model():
    """Loads tokenizer and classification model lazily to reduce memory usage."""
    global _tokenizer, _model, _model_labels
    if _tokenizer is None or _model is None:
        print("DEBUG: Loading news-specific model...")
        _tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/tweet-topic-21-multi")
        _model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/tweet-topic-21-multi"
        ).to("cpu").eval()  # Use eval mode to prevent gradient tracking
        print("DEBUG: Model loaded successfully.")

        # Load actual model labels
        _model_labels = [
            "arts_culture", "business", "environment", "fashion", "finance", "food",
            "health", "lifestyle", "news", "opinion", "politics", "science", "sports",
            "technology", "travel", "wellbeing"
        ]
        print("DEBUG: Model labels loaded:", _model_labels)

def predict_category(article_text, confidence_threshold=0.2):
    """Predicts category using a fine-tuned news classification model."""
    load_model()
    
    if not article_text.strip():
        return "Unknown"
    
    inputs = _tokenizer(article_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = _model(**inputs)
    
    probabilities = F.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_index].item()

    predicted_label = _model_labels[predicted_index]

    print(f"DEBUG: Article classified as '{predicted_label}' with confidence {confidence:.2f}")

    if confidence < confidence_threshold:
        return "Unknown"

    return predicted_label

def simple_summarize(text, max_words=50):
    """Simple text summarization by truncating to max_words."""
    if not text:
        return "No description available."
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def fetch_news():
    """Fetches news articles, classifies them, and returns structured data."""
    try:
        print("DEBUG: Fetching news articles...")
        response = session.get(NEWS_URL, timeout=5)
        response.raise_for_status()
        
        articles = response.json().get("articles", [])
        print(f"DEBUG: {len(articles)} articles fetched.")
        
        categorized_articles = []
        for article in articles:
            title = article.get("title") or "No Title"
            description = article.get("description") or ""
            article_text = f"{title} {description}".strip()
            
            category = predict_category(article_text) if article_text else "Unknown"

            categorized_articles.append({
                "title": title,
                "summary": simple_summarize(description),
                "url": article.get("url", "#"),
                "category": category
            })
        
        return categorized_articles

    except requests.RequestException as e:
        print(f"ERROR: News API request failed - {e}")
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
