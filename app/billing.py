from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

import stripe
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from app.aggregator import User
from app.extensions import db
from app.security import require_csrf
from app.subscriptions import (
    StripeCustomer,
    SubscriptionEntitlement,
    current_user_has_plus,
    get_or_create_entitlement,
)


billing_bp = Blueprint("billing", __name__, template_folder="templates/billing")


def _utcnow():
    return datetime.now(timezone.utc)


def _base_url() -> str:
    env_base = (os.getenv("BASE_URL") or "").strip()
    if env_base:
        return env_base.rstrip("/")
    root = (request.url_root or "").strip()
    return root.rstrip("/") if root else ""


def _stripe_configured() -> bool:
    return bool(os.getenv("STRIPE_SECRET_KEY"))


def _stripe_client():
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    return stripe


def _create_customer_for_user(user: User) -> StripeCustomer | None:
    client = _stripe_client()
    try:
        stripe_customer = client.Customer.create(email=user.email, metadata={"user_id": user.id})
    except Exception as exc:
        current_app.logger.error("Stripe customer create failed: %s", exc)
        return None
    record = StripeCustomer(user_id=user.id, stripe_customer_id=stripe_customer["id"])
    db.session.add(record)
    db.session.commit()
    return record


def _get_or_create_customer(user: User) -> StripeCustomer | None:
    record = StripeCustomer.query.filter_by(user_id=user.id).first()
    if record:
        return record
    return _create_customer_for_user(user)


def _status_from_subscription_state(state: str) -> str:
    state = (state or "").lower()
    if state in {"active", "trialing"}:
        return "active"
    if state in {"past_due", "unpaid"}:
        return "past_due"
    if state in {"canceled", "incomplete_expired"}:
        return "canceled"
    return "none"


def _apply_entitlement(user: User, status: str, period_end_ts: int | None, subscription_id: str | None):
    ent = get_or_create_entitlement(user.id)
    ent.status = status
    ent.stripe_subscription_id = subscription_id or ent.stripe_subscription_id
    if period_end_ts:
        ent.current_period_end = datetime.fromtimestamp(period_end_ts, tz=timezone.utc)
    ent.updated_at = _utcnow()
    db.session.add(ent)
    db.session.commit()
    return ent


def _sync_subscription(user: User, subscription_id: str) -> None:
    client = _stripe_client()
    try:
        subscription = client.Subscription.retrieve(subscription_id, expand=["default_payment_method"])
    except Exception as exc:
        current_app.logger.error("Stripe retrieve subscription failed: %s", exc)
        return
    status = _status_from_subscription_state(subscription.get("status"))
    period_end = subscription.get("current_period_end")
    _apply_entitlement(user, status, period_end, subscription_id)


def _user_for_customer(customer_id: str | None, fallback: Dict[str, Any] | None = None) -> User | None:
    if not customer_id:
        return None
    link = StripeCustomer.query.filter_by(stripe_customer_id=customer_id).first()
    if link:
        return User.query.get(link.user_id)
    email = ((fallback or {}).get("email") or "").strip().lower()
    if email:
        user = User.query.filter_by(email=email).first()
        if user:
            record = StripeCustomer(user_id=user.id, stripe_customer_id=customer_id)
            db.session.add(record)
            db.session.commit()
            return user
    return None


@billing_bp.route("/plus", methods=["GET"])
def plus_page():
    next_url = request.args.get("next") or "/"
    return render_template(
        "billing/plus.html",
        plus_active=current_user_has_plus(),
        next_url=next_url,
    )


@billing_bp.route("/billing/checkout", methods=["POST"])
@login_required
def create_checkout():
    require_csrf()
    if not _stripe_configured():
        return render_template("billing/plus.html", error="Billing is temporarily unavailable.", plus_active=False), 503

    price_id = os.getenv("STRIPE_PRICE_ID_MONTHLY")
    if not price_id:
        return render_template("billing/plus.html", error="Plan not configured.", plus_active=False), 500

    customer = _get_or_create_customer(current_user)
    if not customer:
        return render_template("billing/plus.html", error="Unable to start checkout right now.", plus_active=False), 500

    base = _base_url()
    success_url = f"{base}/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{base}/billing/cancel"

    client = _stripe_client()
    try:
        session = client.checkout.Session.create(
            mode="subscription",
            customer=customer.stripe_customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=False,
        )
    except Exception as exc:
        current_app.logger.error("Checkout session failed: %s", exc)
        return render_template("billing/plus.html", error="Could not start checkout. Try again soon.", plus_active=False), 500

    return redirect(session.url, code=303)


@billing_bp.route("/billing/success", methods=["GET"])
def checkout_success():
    return render_template("billing/success.html", plus_active=current_user_has_plus())


@billing_bp.route("/billing/cancel", methods=["GET"])
def checkout_cancel():
    return render_template("billing/cancel.html", plus_active=current_user_has_plus())


@billing_bp.route("/billing/portal", methods=["GET"])
@login_required
def customer_portal():
    if not _stripe_configured():
        return render_template("billing/plus.html", error="Billing is offline.", plus_active=current_user_has_plus()), 503

    customer = _get_or_create_customer(current_user)
    if not customer:
        return render_template("billing/plus.html", error="Unable to open portal right now.", plus_active=False), 500

    base = _base_url()
    client = _stripe_client()
    try:
        portal = client.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
            return_url=f"{base}/plus",
        )
    except Exception as exc:
        current_app.logger.error("Portal session failed: %s", exc)
        return render_template("billing/plus.html", error="Unable to open billing portal.", plus_active=False), 500

    return redirect(portal.url, code=303)


def _handle_subscription_event(subscription_obj: Dict[str, Any]):
    customer_id = subscription_obj.get("customer")
    user = _user_for_customer(customer_id)
    if not user:
        return
    status = _status_from_subscription_state(subscription_obj.get("status"))
    period_end = subscription_obj.get("current_period_end")
    sub_id = subscription_obj.get("id")
    _apply_entitlement(user, status, period_end, sub_id)


@billing_bp.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        return "Webhook not configured", 400

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        return "Invalid signature", 400
    except Exception as exc:
        current_app.logger.error("Webhook parse failed: %s", exc)
        return "Bad payload", 400

    event_type = event.get("type")
    obj = event.get("data", {}).get("object") or {}

    if event_type == "checkout.session.completed":
        customer_id = obj.get("customer")
        user = _user_for_customer(customer_id, obj.get("customer_details"))
        subscription_id = obj.get("subscription")
        if user and subscription_id:
            _sync_subscription(user, subscription_id)
    elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
        _handle_subscription_event(obj)
    elif event_type == "invoice.payment_succeeded":
        customer_id = obj.get("customer")
        user = _user_for_customer(customer_id)
        if user:
            sub_id = obj.get("subscription")
            period_end = obj.get("current_period_end")
            _apply_entitlement(user, "active", period_end, sub_id)
    elif event_type == "invoice.payment_failed":
        customer_id = obj.get("customer")
        user = _user_for_customer(customer_id)
        if user:
            sub_id = obj.get("subscription")
            _apply_entitlement(user, "past_due", obj.get("current_period_end"), sub_id)

    return jsonify({"received": True})
