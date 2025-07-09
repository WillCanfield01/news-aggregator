from flask import Blueprint, jsonify
from flask_login import login_required
from app.utils.feed_utils import (
    fetch_live_articles,
    get_cached_articles,
    regenerate_summary_for_article
)
from app.utils.bias_utils import bias_bucket

bp = Blueprint("news", __name__, url_prefix="/news")

# Serve main page news (cached for 10 minutes)
@bp.route("/")
def all_news():
    return jsonify(get_cached_articles())

# Refresh articles without using cache (for debug/admin or live testing)
@bp.route("/new")
def new_articles():
    return jsonify(fetch_live_articles())

# Filter news by political bias
@bp.route("/by-bias/<bias>")
def news_by_bias(bias):
    bias = bias.strip().capitalize()
    if bias not in {"Left", "Center", "Right"}:
        return jsonify({"error": "Invalid bias value"}), 400

    filtered = [a for a in get_cached_articles() if bias_bucket(a["bias"]) == bias]
    return jsonify(filtered)

# Filter news by normalized category
@bp.route("/by-category/<category>")
def news_by_category(category):
    normalized = normalize_category(category)
    filtered = [a for a in get_cached_articles() if a["category"] == normalized]
    return jsonify(filtered)

# Regenerate OpenAI summary for a specific article (by ID)
@bp.route("/regenerate-summary/<article_id>")
def regenerate_summary(article_id):
    summary = regenerate_summary_for_article(article_id)
    if summary:
        return jsonify({"summary": summary})
    return jsonify({"error": "Article not found"}), 404