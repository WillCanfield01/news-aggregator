from flask import Blueprint, jsonify
from flask_login import login_required
from app.utils.feed_utils import get_cached_articles, get_new_articles, regenerate_summary_for_article
from app.utils.bias_utils import bias_bucket
from app.utils.category_utils import normalize_category

bp = Blueprint("news", __name__, url_prefix="/news")

@bp.route("/")
def all_news():
    return jsonify(get_cached_articles())

@bp.route("/new")
def new_articles():
    return jsonify(get_new_articles())

@bp.route("/by-bias/<bias>")
def news_by_bias(bias):
    bias = bias.strip().capitalize()
    if bias not in {"Left", "Center", "Right"}:
        return jsonify({"error": "Invalid bias value"}), 400

    filtered = [a for a in get_cached_articles() if bias_bucket(a["bias"]) == bias]
    return jsonify(filtered)

@bp.route("/by-category/<category>")
def news_by_category(category):
    normalized = normalize_category(category)
    filtered = [a for a in get_cached_articles() if a["category"] == normalized]
    return jsonify(filtered)

@bp.route("/regenerate-summary/<article_id>")
def regenerate_summary(article_id):
    summary = regenerate_summary_for_article(article_id)
    if summary:
        return jsonify({"summary": summary})
    return jsonify({"error": "Article not found"}), 404