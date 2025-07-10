from flask import Blueprint, jsonify, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app.utils.feed_utils import get_local_articles_for_user, is_valid_zip
from app import db

bp = Blueprint("local", __name__)

@bp.route("/news/local")
@login_required
def local_news_page():
    return render_template("local_news.html")

@bp.route("/api/news/local")
@login_required
def get_local_news():
    articles = get_local_articles_for_user(current_user)
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
    return redirect(url_for("account_page"))
