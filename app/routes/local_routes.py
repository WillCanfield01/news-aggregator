from flask import Blueprint, jsonify, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from app.utils.feed_utils import is_valid_zip
from app import db
import time

bp = Blueprint("local", __name__)

@bp.route("/news/local")
@login_required
def local_news_page():
    return render_template("local_news.html")

@bp.route("/api/news/local")
@login_required
def get_local_news():
    zip_code = getattr(current_user, "zipcode", None)
    if not zip_code:
        return jsonify([])

    # Ensure cache exists in current_app
    if not hasattr(current_app, "local_articles_cache"):
        current_app.local_articles_cache = {}
    cache = current_app.local_articles_cache

    result = cache.get(zip_code)
    if result and isinstance(result, tuple) and len(result) == 2:
        ts, articles = result
        if time.time() - ts < 15 * 60:
            print(f"Serving {len(articles)} cached local articles for {zip_code}")
            return jsonify(articles)
        else:
            del cache[zip_code]

    # Synchronous fetch to avoid blocking
    from app.utils.geo_utils import fetch_google_local_feed
    try:
        articles = fetch_google_local_feed(zip_code, limit=50)
    except Exception as e:
        print(f"Error fetching articles for {zip_code}: {e}")
        return jsonify([])

    # Deduplicate articles by URL and title
    seen = set()
    deduped_articles = []
    for article in articles:
        url = (article.get("url") or article.get("link") or "").strip()
        title = (article.get("title") or "").strip()
        unique_key = f"{url}|{title}"
        if url and unique_key not in seen:
            seen.add(unique_key)
            deduped_articles.append(article)

    print(f"Articles fetched for {zip_code}: {len(deduped_articles)}")
    cache[zip_code] = (time.time(), deduped_articles)
    return jsonify(deduped_articles)

@bp.route("/update-zipcode", methods=["POST"])
@login_required
def update_zipcode():
    # Accept zip from form or JSON body
    zip_input = request.form.get("zip")
    if zip_input is None:
        json_data = request.get_json(silent=True)
        zip_input = (json_data or {}).get("zip", "")
    zip_input = (zip_input or "").strip()
    if is_valid_zip(zip_input):
        current_user.zipcode = zip_input
        db.session.commit()
        flash("ZIP code updated successfully!", "success")
    else:
        flash("Invalid ZIP code format. Please enter a 5-digit U.S. ZIP.", "error")
    return redirect(url_for("user.account_page"))