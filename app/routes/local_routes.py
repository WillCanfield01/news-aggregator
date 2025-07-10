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

    cache = getattr(current_app, "local_articles_cache", {})

    result = cache.get(zip_code)
    if result:
        ts, articles = result
        if time.time() - ts < 15 * 60:
            print(f"Serving {len(articles)} cached local articles for {zip_code}")
            return jsonify(articles)
        else:
            del cache[zip_code]

    from app.utils.geo_utils import fetch_google_local_feed
    import asyncio
    articles = asyncio.run(fetch_google_local_feed(zip_code, limit=50))
    print(f"Articles fetched for {zip_code}: {len(articles)}")
    cache[zip_code] = (time.time(), articles)
    return jsonify(articles)

@bp.route("/update-zipcode", methods=["POST"])
@login_required
def update_zipcode():
    zip_input = request.form.get("zip") or (request.get_json() or {}).get("zip", "").strip()
    if is_valid_zip(zip_input):
        current_user.zipcode = zip_input
        db.session.commit()
        flash("ZIP code updated successfully!", "success")
    else:
        flash("Invalid ZIP code format. Please enter a 5-digit U.S. ZIP.", "error")
    # Make sure the endpoint matches your user/account route
    return redirect(url_for("user.account_page"))