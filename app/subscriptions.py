from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from flask import current_app, jsonify, request, url_for
from flask_login import current_user

from app.extensions import db


def _utcnow():
    return datetime.now(timezone.utc)


class StripeCustomer(db.Model):
    __tablename__ = "stripe_customers"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)
    stripe_customer_id = db.Column(db.String(120), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow, nullable=False)


class SubscriptionEntitlement(db.Model):
    __tablename__ = "subscription_entitlements"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)
    status = db.Column(db.String(20), default="none", nullable=False, index=True)
    current_period_end = db.Column(db.DateTime(timezone=True), nullable=True)
    stripe_subscription_id = db.Column(db.String(120), nullable=True, index=True)
    updated_at = db.Column(db.DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


def _plus_checkout_url() -> str:
    try:
        from app.plus import get_plus_checkout_url

        return get_plus_checkout_url()
    except Exception:
        try:
            return current_app.config.get("PLUS_CHECKOUT_URL", "/billing/checkout") or "/billing/checkout"
        except Exception:
            return "/billing/checkout"


def current_user_is_plus() -> bool:
    """
    Return True when the logged-in user has an active Plus entitlement.
    Falls back to a user.is_plus flag or a SubscriptionEntitlement row.
    """
    if not getattr(current_user, "is_authenticated", False):
        return False
    try:
        flag = getattr(current_user, "is_plus", None)
        if isinstance(flag, bool) and flag:
            return True
    except Exception:
        pass

    ent = SubscriptionEntitlement.query.filter_by(user_id=current_user.id).first()
    if not ent or ent.status != "active":
        return False
    if ent.current_period_end and ent.current_period_end < _utcnow():
        return False
    return True


def current_user_has_plus() -> bool:
    """Backward-compatible wrapper for templates/routes still calling the old helper."""
    return current_user_is_plus()


def get_or_create_entitlement(user_id: int) -> SubscriptionEntitlement:
    ent = SubscriptionEntitlement.query.filter_by(user_id=user_id).first()
    if ent:
        return ent
    ent = SubscriptionEntitlement(user_id=user_id, status="none")
    db.session.add(ent)
    db.session.commit()
    return ent


def plus_required(view_func: Callable):
    """
    Decorator for API endpoints that require Plus.
    Returns a 402 JSON payload with upgrade guidance instead of redirecting.
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if current_user_is_plus():
            return view_func(*args, **kwargs)

        payload = {
            "ok": False,
            "error": "plus_required",
            "reason": "plus_required",
            "title": "Want to keep playing?",
            "message": "Plus unlocks unlimited plays, streak protection, and past days.",
            "next_action": "upgrade_plus",
            "checkout_url": _plus_checkout_url(),
        }
        try:
            payload["login_url"] = url_for("auth.auth_page", next=request.url)
        except Exception:
            payload["login_url"] = "/auth"
        return jsonify(payload), 402

    return wrapper
