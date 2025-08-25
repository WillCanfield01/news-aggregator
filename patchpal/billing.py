# patchpal/billing.py
from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Optional

import stripe
from flask import Blueprint, request, jsonify
from slack_sdk import WebClient

from .storage import SessionLocal, Workspace

# --- Config ---
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
PRICE_ID = os.getenv("STRIPE_PRICE_ID")
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://example.com")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
slack_client = WebClient(token=SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None

billing_bp = Blueprint("billing_bp", __name__)

# ---------- Core helpers ----------

def ensure_trial(ws: Workspace, db) -> None:
    if not ws.trial_ends_at:
        ws.trial_ends_at = datetime.utcnow() + timedelta(days=14)
        ws.plan = ws.plan or "trial"
        db.commit()

def is_active(team_id: Optional[str] = None, ws: Optional[Workspace] = None) -> bool:
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

def portal_url(ws: Workspace) -> Optional[str]:
    """Create a Customer Portal session (requires ws.customer_id)."""
    if not stripe.api_key or not ws.customer_id:
        return None
    sess = stripe.billing_portal.Session.create(
        customer=ws.customer_id,
        return_url=f"{APP_BASE_URL}/billing/success",
    )
    return sess.url

def get_next_renewal(ws: Workspace) -> Optional[datetime]:
    """Return next renewal (current_period_end) for pro subs, UTC."""
    if not stripe.api_key or not ws.subscription_id:
        return None
    sub = stripe.Subscription.retrieve(ws.subscription_id)
    ts = int(sub.get("current_period_end", 0))
    return datetime.utcfromtimestamp(ts) if ts else None

# ---------- DMs / notifications ----------

def _dm(slack: WebClient, user_id: Optional[str], channel_id: Optional[str], text: str):
    try:
        if slack is None:
            return
        if user_id:
            im = slack.conversations_open(users=user_id)
            cid = im["channel"]["id"]
            slack.chat_postMessage(channel=cid, text=text)
        elif channel_id:
            slack.chat_postMessage(channel=channel_id, text=text)
    except Exception:
        pass

def dm_trial_or_checkout(slack: WebClient, ws: Workspace) -> None:
    """Inactive: send daily upgrade DM with Checkout link."""
    now = datetime.utcnow()
    if ws.last_billing_nag and (now - ws.last_billing_nag).days < 1:
        return
    url = checkout_url(ws.team_id)
    text = (
        "⏳ *PatchPal trial ended* — posts are paused.\n"
        f"Upgrade to resume daily Top 5: {url}\n"
        "_$9/workspace/mo_"
    )
    _dm(slack, ws.contact_user_id, ws.post_channel, text)
    with SessionLocal() as db:
        w = db.query(Workspace).filter_by(id=ws.id).first()
        if w:
            w.last_billing_nag = now
            db.commit()

def dm_trial_ending_soon(slack: WebClient, ws: Workspace) -> None:
    """3-day pre-expiry reminder (rate-limited to once/day)."""
    if not ws.trial_ends_at:
        return
    now = datetime.utcnow()
    days_left = (ws.trial_ends_at - now).days
    if days_left != 3:
        return
    if ws.last_trial_warn and (now - ws.last_trial_warn).days < 1:
        return
    url = checkout_url(ws.team_id)
    text = (
        "🔔 *Your PatchPal trial ends in ~3 days.*\n"
        f"Keep the daily Top 5 coming: {url}"
    )
    _dm(slack, ws.contact_user_id, ws.post_channel, text)
    with SessionLocal() as db:
        w = db.query(Workspace).filter_by(id=ws.id).first()
        if w:
            w.last_trial_warn = now
            db.commit()

def dm_payment_failed(slack: WebClient, ws: Workspace) -> None:
    """Warn on renewal failure (rate-limited to once/day)."""
    now = datetime.utcnow()
    if ws.last_payment_fail_nag and (now - ws.last_payment_fail_nag).days < 1:
        return
    url = portal_url(ws) or checkout_url(ws.team_id)
    text = (
        "⚠️ *PatchPal payment failed.* Posts will pause if not resolved.\n"
        f"Update payment details here: {url}"
    )
    _dm(slack, ws.contact_user_id, ws.post_channel, text)
    with SessionLocal() as db:
        w = db.query(Workspace).filter_by(id=ws.id).first()
        if w:
            w.last_payment_fail_nag = now
            db.commit()

# ---------- Webhook ----------

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
            cust_id = obj.get("customer")
            if team_id:
                ws = db.query(Workspace).filter_by(team_id=team_id).first()
                if ws:
                    ws.plan = "pro"
                    ws.paid_at = datetime.utcnow()
                    ws.subscription_id = sub_id
                    if cust_id:
                        ws.customer_id = cust_id
                    db.commit()

        elif etype in {"customer.subscription.deleted", "customer.subscription.canceled"}:
            sub_id = obj.get("id")
            if sub_id:
                ws = db.query(Workspace).filter_by(subscription_id=sub_id).first()
                if ws:
                    ws.plan = "canceled"
                    db.commit()

        elif etype == "invoice.payment_failed":
            # Look up workspace by subscription; send a polite DM with portal link.
            sub_id = obj.get("subscription")
            if sub_id:
                ws = db.query(Workspace).filter_by(subscription_id=sub_id).first()
                if ws:
                    dm_payment_failed(slack_client, ws)

    return jsonify(received=True)

# Tiny success/cancel landing
@billing_bp.route("/success")
def success_page():
    return "Thanks! Your subscription is active. You can close this tab."

@billing_bp.route("/cancel")
def cancel_page():
    return "No worries—your trial remains. You can upgrade anytime from Slack."
