from flask import Blueprint, jsonify, render_template, request
from flask_login import login_required, current_user
from app.utils.feed_utils import (
    get_cached_articles,
    get_new_articles,
    manual_refresh_articles,
    regenerate_summary_for_article,
    normalize_category,
    get_local_articles_for_user,
)
from app.utils.bias_utils import bias_bucket

bp = Blueprint('news', __name__)

# Main homepage (serves index.html)
@bp.route('/')
def home():
    return render_template('index.html')

# Main news API (returns cached articles)
@bp.route('/news')
def get_news():
    return jsonify(get_cached_articles())

# Manual refresh endpoint (admin/testing)
@bp.route('/refresh')
def manual_refresh():
    result = manual_refresh_articles()
    return jsonify(result)

# Get most recently refreshed articles (new batch)
@bp.route('/new')
def new_articles():
    return jsonify(get_new_articles())

# Regenerate OpenAI summary for a specific article (by ID)
@bp.route('/regenerate-summary/<article_id>')
def regenerate_summary(article_id):
    summary = regenerate_summary_for_article(article_id)
    if summary:
        return jsonify({"summary": summary})
    return jsonify({"error": "Article not found"}), 404

# Filter by political bias bucket
@bp.route('/news/by-bias/<bias>')
def news_by_bias(bias):
    bias = bias.strip().capitalize()
    if bias not in {"Left", "Center", "Right"}:
        return jsonify({"error": "Invalid bias value"}), 400
    filtered = [a for a in get_cached_articles() if bias_bucket(a["bias"]) == bias]
    return jsonify(filtered)

# Filter by category
@bp.route('/news/by-category/<category>')
def news_by_category(category):
    normalized = normalize_category(category)
    filtered = [a for a in get_cached_articles() if a["category"] == normalized]
    return jsonify(filtered)