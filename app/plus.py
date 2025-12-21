import os
from flask import current_app
from flask_login import current_user


def get_plus_checkout_url() -> str:
    try:
        url = current_app.config.get("PLUS_CHECKOUT_URL") or "/billing/checkout"
    except Exception:
        url = os.getenv("STRIPE_PLUS_CHECKOUT_URL") or "/billing/checkout"
    return url


def is_plus_user() -> bool:
    """
    Return True if the current user has Plus; placeholder uses existing helper when available.
    """
    try:
        if hasattr(current_user, "is_authenticated") and current_user.is_authenticated:
            from app.subscriptions import current_user_has_plus

            return current_user_has_plus()
    except Exception:
        pass
    return False  # TODO: wire to real entitlement if/when available
