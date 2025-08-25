# patchpal/billing.py
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Optional

import stripe
from flask import Blueprint, request, jsonify

from .storage import SessionLocal, Workspace

# --- Stripe config ---
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
PRICE_ID = os.getenv("STRIPE_PRICE_ID")                 # e.g. price_123
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://example.com")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

billing_bp = Blueprint("billing_bp", __name__)

# -------- Core helpers --------

def ensure_trial(ws: Workspace, db) -> None:
    """Set a 14-day trial if missing."""
    if not ws.trial_ends_at:
        ws.trial_ends_at = datetime.utcnow() + timedelta(days=14)
        ws.plan = ws.plan or "trial"
        db.commit()

def is_active(team_id: Optional[str] = None, ws: Optional[Workspace] = None) -> bool:
    """Active if paid, or trial not expired."""
    if not ws:
        with SessionLocal() as db:
            ws = db.query(Workspace).filter_by(team_id=team_id).first()
    if not ws:
        return False
    if ws.plan == "pro" and ws.paid_at:
        return True
    if ws.trial_ends_at and datetime.utcnow() <= ws.trial_ends_at:
        return True
    return False

def checkout_url(team_id: str) -> str:
    """Create a Stripe Checkout session and return its URL."""
    if not stripe.api_key or not PRICE_ID:
        return f"{APP_BASE_URL}/billing/upgrade-not-configured"

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": PRICE_ID, "quantity": 1}],
        allow_promotion_codes=True,
        client_reference_id=team_id,
        metadata={"team_id": team_id},
        success_url=f"{APP_BASE_URL}/billing/success?team_id={team_id}&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{APP_BASE_URL}/billing/cancel?team_id={team_id}",
    )
    return session.url

def dm_trial_or_checkout(slack_client, ws: Workspace) -> None:
    """DM an admin/contact with an upgrade link. Rate-limited to once/day via last_billing_nag."""
    now = datetime.utcnow()
    if ws.last_billing_nag and (now - ws.last_billing_nag).days < 1:
        return

    url = checkout_url(ws.team_id)
    text = (
        "⏳ *PatchPal trial ended* — posts are paused.\n"
        f"Click to upgrade and resume daily Top 5 posts: {url}\n"
        "_$9/workspace/mo · 14-day free trial included for new installs_"
    )

    try:
        if ws.contact_user_id:
            im = slack_client.conversations_open(users=ws.contact_user_id)
            channel_id = im["channel"]["id"]
            slack_client.chat_postMessage(channel=channel_id, text=text)
        elif ws.post_channel:
            slack_client.chat_postMessage(channel=ws.post_channel, text=text)
    except Exception:
        pass

    with SessionLocal() as db:
        w = db.query(Workspace).filter_by(id=ws.id).first()
        if w:
            w.last_billing_nag = now
            db.commit()

# -------- Webhook --------

@billing_bp.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    if not WEBHOOK_SECRET:
        return ("Webhook secret not set", 400)

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    except Exception as e:
        return (f"Invalid signature: {e}", 400)

    etype = event.get("type")
    obj = event.get("data", {}).get("object", {})

    with SessionLocal() as db:
        if etype == "checkout.session.completed":
            team_id = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("team_id")
            sub_id = obj.get("subscription")
            if team_id:
                ws = db.query(Workspace).filter_by(team_id=team_id).first()
                if ws:
                    ws.plan = "pro"
                    ws.paid_at = datetime.utcnow()
                    ws.subscription_id = sub_id
                    db.commit()

        elif etype in {"customer.subscription.deleted", "customer.subscription.canceled"}:
            sub_id = obj.get("id")
            if sub_id:
                ws = db.query(Workspace).filter_by(subscription_id=sub_id).first()
                if ws:
                    ws.plan = "canceled"
                    db.commit()

    return jsonify(received=True)

@billing_bp.route("/success")
def success_page():
    return "Thanks! Your subscription is active. You can close this tab."

@billing_bp.route("/cancel")
def cancel_page():
    return "No worries—your trial remains. You can upgrade anytime from Slack."
