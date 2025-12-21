from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from flask import Blueprint, redirect, render_template, request, url_for, session, jsonify
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import or_, func

from app.aggregator import User
from app.email_utils import send_email
from app.extensions import db
from app.security import require_csrf


auth_bp = Blueprint("auth", __name__, template_folder="templates/auth")

MAGIC_LINK_TTL_MINUTES = 15
REQUEST_COOLDOWN_SECONDS = 60

# Lightweight in-memory throttling (per-process)
_ip_last_request: dict[str, datetime] = {}
_email_last_request: dict[str, datetime] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MagicLinkToken(db.Model):
    __tablename__ = "magic_link_tokens"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    token_hash = db.Column(db.String(128), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    used_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow, nullable=False)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)


def _base_url() -> str:
    env_base = (os.getenv("BASE_URL") or "").strip()
    if env_base:
        return env_base.rstrip("/")
    root = (request.url_root or "").strip()
    return root.rstrip("/") if root else ""


def _hash_token(raw: str) -> str:
    secret = os.getenv("MAGIC_LINK_SECRET", "")
    return hashlib.sha256(f"{raw}{secret}".encode("utf-8")).hexdigest()


def _is_rate_limited(email: str, ip_addr: str | None) -> bool:
    now = _utcnow()
    cutoff = now - timedelta(seconds=REQUEST_COOLDOWN_SECONDS)
    recent_email = _email_last_request.get(email)
    recent_ip = _ip_last_request.get(ip_addr or "")
    if (recent_email and recent_email > cutoff) or (recent_ip and recent_ip > cutoff):
        return True
    # DB-level throttle
    try:
        recent_db = (
            MagicLinkToken.query.filter(
                MagicLinkToken.expires_at > now,
                MagicLinkToken.created_at > cutoff,
                or_(MagicLinkToken.ip_address == ip_addr, func.lower(User.email) == email),
            )
            .join(User, MagicLinkToken.user_id == User.id)
            .first()
        )
    except Exception:
        recent_db = None
    return recent_db is not None


def _mark_request(email: str, ip_addr: str | None):
    now = _utcnow()
    _email_last_request[email] = now
    _ip_last_request[ip_addr or ""] = now


def _safe_next_url(next_param: str | None) -> str:
    if not next_param:
        return "/"
    if next_param.startswith("/"):
        return next_param
    return "/"


def _send_magic_link(to_email: str, token: str, next_url: str):
    base = _base_url()
    params = {"token": token}
    if next_url and next_url != "/":
        params["next"] = next_url
    verify_url = f"{base}/auth/verify?{urlencode(params)}"
    subject = "Your Roundup magic link"
    html = f"""
        <p>Click the secure link to sign in. This link expires in {MAGIC_LINK_TTL_MINUTES} minutes.</p>
        <p><a href="{verify_url}">Sign in to The Roundup</a></p>
        <p>If you did not request this, you can ignore the email.</p>
    """
    text = (
        f"Use this link to sign in (expires in {MAGIC_LINK_TTL_MINUTES} minutes): {verify_url}\n\n"
        "If you didn't request this, you can ignore the email."
    )
    send_email(to_email, subject, html, text)


@auth_bp.route("/auth", methods=["GET"])
def auth_page():
    next_url = request.args.get("next") or "/"
    sent = request.args.get("sent") == "1"
    return render_template("auth/login.html", next_url=_safe_next_url(next_url), sent=sent)


@auth_bp.route("/auth/request-link", methods=["POST"])
def request_magic_link():
    require_csrf()
    email = (request.form.get("email") or (request.get_json(silent=True) or {}).get("email") or "").strip().lower()
    next_url = _safe_next_url(request.form.get("next") or (request.get_json(silent=True) or {}).get("next"))
    ip_addr = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr
    user_agent = (request.headers.get("User-Agent") or "")[:250]

    if not email:
        return render_template("auth/login.html", error="Email is required.", next_url=next_url), 400

    if _is_rate_limited(email, ip_addr):
        return render_template("auth/login.html", error="Check your inbox. You can request a new link in a minute.", next_url=next_url), 429

    user = User.query.filter(func.lower(User.email) == email).first()
    if not user:
        user = User(email=email, username=None, is_confirmed=True, is_active=True)
        db.session.add(user)
        db.session.commit()

    raw_token = secrets.token_urlsafe(32)
    hashed = _hash_token(raw_token)
    expires = _utcnow() + timedelta(minutes=MAGIC_LINK_TTL_MINUTES)
    magic_row = MagicLinkToken(
        user_id=user.id,
        token_hash=hashed,
        expires_at=expires,
        ip_address=ip_addr,
        user_agent=user_agent,
    )
    db.session.add(magic_row)
    db.session.commit()

    _mark_request(email, ip_addr)
    _send_magic_link(email, raw_token, next_url)

    if request.is_json:
        return jsonify({"ok": True}), 200
    params = {"sent": "1"}
    if next_url:
        params["next"] = next_url
    return redirect(url_for("auth.auth_page", **params))


@auth_bp.route("/auth/verify", methods=["GET"])
def verify_magic_link():
    raw_token = (request.args.get("token") or "").strip()
    next_url = _safe_next_url(request.args.get("next"))
    if not raw_token:
        return render_template("auth/login.html", error="Invalid or missing token."), 400

    hashed = _hash_token(raw_token)
    now = _utcnow()
    token_row = MagicLinkToken.query.filter_by(token_hash=hashed).first()
    if not token_row or token_row.used_at or token_row.expires_at < now:
        return render_template("auth/login.html", error="This link is no longer valid."), 400

    user = User.query.get(token_row.user_id)
    if not user or not user.is_active:
        return render_template("auth/login.html", error="Account is inactive."), 400

    token_row.used_at = now
    user.last_login_at = now
    if not user.is_confirmed:
        user.is_confirmed = True
    db.session.add(token_row)
    db.session.add(user)
    db.session.commit()

    login_user(user, remember=True)
    return redirect(next_url or "/")


@auth_bp.route("/auth/logout", methods=["POST"])
@login_required
def logout():
    require_csrf()
    logout_user()
    session.pop("_csrf_token", None)
    return redirect("/")
