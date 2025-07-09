from flask import Blueprint, jsonify
from flask_login import login_required, current_user
from app.models import SavedArticle
from app import db

bp = Blueprint("user", __name__, url_prefix="/user")

@bp.route('/me')
@login_required
def me():
    return jsonify({
        'id': current_user.id,
        'username': current_user.username,
        'email': current_user.email,
        'zipcode': current_user.zipcode
    })

@bp.route("/account")
@login_required
def account():
    saved = SavedArticle.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        "username": current_user.username,
        "saved_articles": [
            {
                "id": a.article_id,
                "title": a.title,
                "url": a.url,
                "summary": a.summary,
                "source": a.source,
                "category": a.category
            } for a in saved
        ]
    })

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