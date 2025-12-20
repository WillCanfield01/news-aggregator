from __future__ import annotations

import os
import secrets
import threading
import time

from flask import Blueprint, abort, jsonify, make_response, render_template, request, session, url_for

from app.roulette.admin_jobs import enqueue_job, get_running_job

# Admin panel lives at /admin/roulette
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# ---- Auth helpers ---------------------------------------------------------
try:
    from flask_login import current_user  # type: ignore

    _HAS_LOGIN = True
except Exception:  # pragma: no cover
    class _Anon:
        is_authenticated = False
        is_admin = False

    current_user = _Anon()  # type: ignore
    _HAS_LOGIN = False


def _require_admin():
    """
    Option A: Use Flask-Login when available, requiring an authenticated admin.
    Option B: Fallback to basic auth with ADMIN_USER/ADMIN_PASS if no user system exists.
    """
    if _HAS_LOGIN:
        if getattr(current_user, "is_authenticated", False) and getattr(current_user, "is_admin", False):
            return
        abort(403)

    admin_user = os.getenv("ADMIN_USER")
    admin_pass = os.getenv("ADMIN_PASS")
    auth = request.authorization
    if admin_user and admin_pass and auth and auth.username == admin_user and auth.password == admin_pass:
        return
    resp = make_response("Unauthorized", 401)
    resp.headers["WWW-Authenticate"] = 'Basic realm="Admin"'
    abort(resp)


# ---- CSRF helpers ---------------------------------------------------------
_CSRF_KEY = "roulette_admin_csrf"


def _get_or_set_csrf() -> str:
    token = session.get(_CSRF_KEY)
    if not token:
        token = secrets.token_urlsafe(16)
        session[_CSRF_KEY] = token
    return token


def _validate_csrf(token: str | None) -> bool:
    expected = session.get(_CSRF_KEY)
    if not expected or not token:
        return False
    try:
        return secrets.compare_digest(expected, token)
    except Exception:
        return False


# ---- Rate limit guard -----------------------------------------------------
_last_regen_ts = 0.0
_rate_lock = threading.Lock()
_REGEN_COOLDOWN_SECONDS = 60


def _check_rate_limit() -> tuple[bool, float]:
    now = time.time()
    with _rate_lock:
        global _last_regen_ts
        if now - _last_regen_ts < _REGEN_COOLDOWN_SECONDS:
            return False, _REGEN_COOLDOWN_SECONDS - (now - _last_regen_ts)
    return True, 0.0


# ---- Routes ---------------------------------------------------------------
@admin_bp.after_request
def _noindex(resp):
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    return resp


@admin_bp.get("/roulette")
def admin_roulette_page():
    _require_admin()
    csrf_token = _get_or_set_csrf()
    return make_response(render_template("admin/roulette.html", csrf_token=csrf_token))


@admin_bp.post("/roulette/regen")
def admin_roulette_regen():
    _require_admin()
    payload = request.get_json(silent=True) if request.is_json else None
    csrf_token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or (payload or {}).get("csrf_token")
    if not _validate_csrf(csrf_token):
        return jsonify({"error": "invalid_csrf"}), 400

    running = get_running_job()
    if running:
        return jsonify({"error": "already_running", "job_id": running.id}), 409

    ok, retry_in = _check_rate_limit()
    if not ok:
        return jsonify({"error": "rate_limited", "retry_in_seconds": int(retry_in)}), 429

    job = enqueue_job(force=1, requested_ip=request.headers.get("X-Forwarded-For", request.remote_addr))
    with _rate_lock:
        global _last_regen_ts
        _last_regen_ts = time.time()
    return (
        jsonify(
            {
                "started": True,
                "job_id": job.id,
                "status_url": url_for("roulette.regen_status", job_id=job.id, _external=True),
            }
        ),
        202,
    )
