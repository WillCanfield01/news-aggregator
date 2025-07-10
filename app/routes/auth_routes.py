from flask import Blueprint, request, jsonify, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from flask import current_app
from sqlalchemy.exc import OperationalError
import time
from app.models import User
from app import db, login_manager
from app.utils.email_utils import generate_confirmation_token, confirm_token, send_confirmation_email

bp = Blueprint('auth', __name__)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@bp.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        # Validate login...
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@bp.route("/signup", methods=["POST"])
def signup():
    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip().lower()
    zipcode_input = data.get("zipcode", "").strip()

    if not username or not password or not email:
        return jsonify({"error": "All fields are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    user = User(username=username, email=email, zipcode=zipcode_input)
    user.set_password(password)
    try:
        db.session.add(user)
        db.session.commit()
        token = generate_confirmation_token(user.email)
        send_confirmation_email(user.email, username, token)
        return jsonify({"success": True, "message": "Signup complete! Check your email to confirm."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Signup failed. Please try again."}), 500

@bp.route("/confirm/<token>")
def confirm_email(token):
    email = confirm_token(token)
    if not email:
        return "The confirmation link is invalid or has expired.", 400

    user = User.query.filter_by(email=email).first_or_404()
    if user.is_confirmed:
        return "Account already confirmed. Please login.", 200
    else:
        user.is_confirmed = True
        db.session.commit()
        return redirect("https://therealroundup.com/?confirmed=true")
    
@login_manager.user_loader
def load_user(user_id):
    retries = 3
    for attempt in range(retries):
        try:
            return User.query.get(int(user_id))
        except OperationalError as e:
            current_app.logger.warning(f"DB connection failed: {e}")
            time.sleep(2)  # wait a moment and retry
    return None  # gracefully fail if Neon isn't awake