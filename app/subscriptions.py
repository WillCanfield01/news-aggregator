from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
from typing import Callable

from flask import redirect, request, url_for
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


def current_user_has_plus() -> bool:
    """Return True when the logged-in user has an active, non-expired Plus entitlement."""
    if not getattr(current_user, "is_authenticated", False):
        return False
    ent = SubscriptionEntitlement.query.filter_by(user_id=current_user.id).first()
    if not ent or ent.status != "active":
        return False
    if ent.current_period_end and ent.current_period_end < _utcnow():
        return False
    return True


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
    Decorator to require authentication + Plus entitlement.
    Redirects to login first, then Plus landing.
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not getattr(current_user, "is_authenticated", False):
            return redirect(url_for("auth.auth_page", next=request.url))
        if not current_user_has_plus():
            return redirect(url_for("billing.plus_page", next=request.url))
        return view_func(*args, **kwargs)

    return wrapper
