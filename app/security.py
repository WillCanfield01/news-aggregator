import secrets
from flask import session, request, abort


def generate_csrf_token() -> str:
    """
    Simple session-backed CSRF token helper.
    Returns the existing token or creates one if missing.
    """
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(16)
        session["_csrf_token"] = token
    return token


def require_csrf() -> None:
    """
    Validate CSRF token from form field, JSON body, or header.
    Raises 400 on failure.
    """
    session_token = session.get("_csrf_token")
    form_token = request.form.get("csrf_token")
    json_token = None
    if request.is_json:
        try:
            json_token = (request.get_json(silent=True) or {}).get("csrf_token")
        except Exception:
            json_token = None
    header_token = request.headers.get("X-CSRF-Token")
    supplied = form_token or json_token or header_token
    if not session_token or not supplied or supplied != session_token:
        abort(400, description="Invalid CSRF token")
