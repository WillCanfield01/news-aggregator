from flask import Blueprint, jsonify, render_template
from flask_login import login_required, current_user
from app.models import SavedArticle
from app import db

bp = Blueprint("user", __name__, url_prefix="/user")

@bp.route("/account")
@login_required
def account_page():
    saved = SavedArticle.query.filter_by(user_id=current_user.id).all()
    return render_template("account.html", username=current_user.username, saved_articles=saved)

@bp.route("/saved-articles")
@login_required
def saved_articles():
    saved = SavedArticle.query.filter_by(user_id=current_user.id).all()
    return jsonify([
        {
            "id": a.article_id,
            "title": a.title,
            "url": a.url,
            "summary": a.summary,
            "source": a.source,
            "category": a.category
        } for a in saved
    ])

@bp.route("/me")
@login_required
def get_user_profile():
    return jsonify({"username": current_user.username})