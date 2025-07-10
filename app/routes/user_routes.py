from flask import Blueprint, jsonify, render_template, request
from flask_login import login_required, current_user
from app.models import SavedArticle
from app import db

bp = Blueprint("user", __name__)  # Remove url_prefix for exact match

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

@bp.route("/save-article", methods=["POST"])
@login_required
def save_article():
    data = request.get_json() or {}
    article_id = data.get("id")
    title = data.get("title")
    url = data.get("url")
    summary = data.get("summary")
    source = data.get("source")
    category = data.get("category")

    if SavedArticle.query.filter_by(user_id=current_user.id, article_id=article_id).first():
        return jsonify({"error": "Article already saved"}), 400

    current_saved_count = SavedArticle.query.filter_by(user_id=current_user.id).count()
    if current_saved_count >= 10:
        return jsonify({"error": "Save limit reached (10 articles max). Please unsave one first."}), 403

    saved = SavedArticle(
        user_id=current_user.id,
        article_id=article_id,
        title=title,
        url=url,
        summary=summary,
        source=source,
        category=category
    )
    db.session.add(saved)
    db.session.commit()
    return jsonify({"success": True, "message": "Article saved"})

@bp.route("/save", methods=["POST"])
@login_required
def alias_save_article():
    return save_article()

@bp.route("/unsave-article", methods=["POST"])
@login_required
def unsave_article():
    data = request.get_json() or {}
    article_id = data.get("id")
    saved = SavedArticle.query.filter_by(user_id=current_user.id, article_id=article_id).first()
    if not saved:
        return jsonify({"error": "Article not found in saved list"}), 404

    db.session.delete(saved)
    db.session.commit()
    return jsonify({"success": True, "message": "Article unsaved"})

@bp.route("/reset-password", methods=["POST"])
@login_required
def reset_password():
    data = request.get_json() or {}
    current = data.get("current_password", "").strip()
    new = data.get("new_password", "").strip()

    if not current_user.check_password(current):
        return jsonify({"error": "Current password is incorrect"}), 400
    if len(new) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400

    current_user.set_password(new)
    db.session.commit()
    return jsonify({"success": True, "message": "Password updated successfully"})