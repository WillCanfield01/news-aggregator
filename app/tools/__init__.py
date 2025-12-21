from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from flask import Blueprint, abort, jsonify, redirect, render_template, request, url_for, session, current_app
try:
    from flask_login import current_user  # type: ignore
except Exception:  # pragma: no cover
    class _Anon:
        is_authenticated = False
        id = None
    current_user = _Anon()  # type: ignore

from app.extensions import db
from app.tools.handlers import (
    run_expense_splitter,
    run_resume_bullets,
    run_trip_planner,
    run_daily_phrase,
    run_decision_helper,
    run_worth_it,
    run_social_post_polisher,
    run_grocery_list_create,
    run_grocery_list_get,
    run_grocery_list_update,
    run_countdown_create_share,
    run_countdown_get,
)
from app.tools.models import SharedExpenseEvent, default_expiry
from app.tools.registry import get_enabled_tools, get_tool_by_slug
from app.subscriptions import current_user_has_plus
import logging
import string
import random

# UI blueprint
tools_bp = Blueprint(
    "tools",
    __name__,
    url_prefix="/tools",
    template_folder="../templates/tools",
)

# API blueprint for tool execution
api_tools_bp = Blueprint(
    "tools_api",
    __name__,
    url_prefix="/api/tools",
)

# Safety guard for request payload size
MAX_INPUT_CHARS = 10_000


def _base_url() -> str:
    """Return the canonical base URL from env or the current request root."""
    env_base = (os.environ.get("BASE_URL") or "").strip()
    if env_base:
        return env_base.rstrip("/")
    root = (request.url_root or "").strip()
    return root.rstrip("/") if root else ""


def _grocery_share_url(token: str) -> str:
    """Build an absolute grocery share URL."""
    base = _base_url()
    if not base:
        return url_for("tools.grocery_share_view", token=token, _external=True)
    return f"{base}/tools/grocery/{token}"


def _aggregate_string_length(payload: Any) -> int:
    """Recursively count characters across all string fields in payload."""
    if isinstance(payload, str):
        return len(payload)
    if isinstance(payload, dict):
        return sum(_aggregate_string_length(v) for v in payload.values())
    if isinstance(payload, (list, tuple, set)):
        return sum(_aggregate_string_length(v) for v in payload)
    return 0


def _build_response(ok: bool, *, data: Any, error: dict[str, Any] | None, request_id: str):
    return jsonify({"ok": ok, "error": error, "data": data, "request_id": request_id})


def _error_response(code: str, message: str, request_id: str, status: int = 400):
    return _build_response(False, data=None, error={"code": code, "message": message}, request_id=request_id), status


def _ref_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _current_user_id() -> int | None:
    if getattr(current_user, "is_authenticated", False):
        try:
            return int(current_user.id)
        except Exception:
            return None
    return None


def _free_share_allowed() -> bool:
    if current_user_has_plus():
        return True
    user_id = _current_user_id()
    if user_id:
        owned = SharedExpenseEvent.query.filter_by(user_id=user_id).count()
        if owned >= 1:
            return False
    free_tokens = session.get("free_share_tokens") or []
    return len(free_tokens) < 1


def _record_free_share(token: str):
    if current_user_has_plus():
        return
    user_id = _current_user_id()
    if user_id:
        return
    free_tokens = session.get("free_share_tokens") or []
    free_tokens.append(token)
    session["free_share_tokens"] = free_tokens[-3:]


def _share_limit_response(request_id: str):
    return _error_response(
        "PLUS_REQUIRED",
        "Upgrade to Roundup Plus to create more share links.",
        request_id,
        status=402,
    )


def _build_response(ok: bool, *, data: Any, error: dict[str, Any] | None, request_id: str):
    return jsonify({"ok": ok, "error": error, "data": data, "request_id": request_id})


def _error_response(code: str, message: str, request_id: str, status: int = 400):
    return _build_response(False, data=None, error={"code": code, "message": message}, request_id=request_id), status


def _validate_input_against_schema(tool: Dict[str, Any], payload: Dict[str, Any], request_id: str):
    """Validate input payload against the tool schema and enforce per-field limits."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    cleaned: Dict[str, Any] = {}
    for field in tool.get("inputs", []):
        name = field.get("name")
        field_type = field.get("type")
        required = field.get("required", False)
        max_chars = field.get("max_chars")
        options = field.get("options") or []
        default_value = field.get("default")

        incoming = payload.get(name)

        if incoming is None or incoming == "":
            if default_value is not None and default_value != "":
                incoming = str(default_value)
            elif required:
                raise ValueError(f'"{name}" is required.')
            else:
                incoming = ""

        if field_type in {"text", "textarea", "select"}:
            if not isinstance(incoming, str):
                raise ValueError(f'"{name}" must be a string.')
            if max_chars and len(incoming) > max_chars:
                raise ValueError(f'"{name}" is too long (limit {max_chars} characters).')
            if field_type == "select":
                if incoming == "":
                    incoming = default_value or ""
                if incoming and options and incoming not in options:
                    raise ValueError(f'"{name}" must be one of: {", ".join(options)}.')
        cleaned[name] = incoming.strip() if isinstance(incoming, str) else incoming

    total_chars = _aggregate_string_length(cleaned)
    if total_chars > MAX_INPUT_CHARS:
        raise ValueError(f"Input is too large. Limit is {MAX_INPUT_CHARS} characters across text fields.")

    return cleaned


@tools_bp.route("/", methods=["GET"])
@tools_bp.route("", methods=["GET"])
def tools_home():
    enabled_tools = get_enabled_tools()
    cfg = getattr(current_app, "config", {}) or {}
    show_plus_upsell = bool(cfg.get("SHOW_PLUS_UPSELL"))
    return render_template(
        "tools/index.html",
        tools=enabled_tools,
        show_plus_upsell=show_plus_upsell,
    )


@tools_bp.route("/expense-splitter/<token>", methods=["GET"])
def expense_splitter_view(token: str):
    tool = get_tool_by_slug("expense-splitter")
    if not tool:
        abort(404)

    event = SharedExpenseEvent.query.filter_by(token=token).first()
    now = datetime.now(timezone.utc)
    if not event or not event.expires_at or event.expires_at < now:
        return render_template(
            "tools/tool_page.html",
            tool=tool,
            error_message="This shared expense could not be found or has expired.",
            view_mode=True,
        ), 404

    payload = event.to_payload()
    if payload.get("type") == "trip-planner":
        return trip_share_view(token)

    saved_output = payload.get("settlement_text", "")
    saved_input = {
        "event_name": payload.get("event_name", ""),
        "participants": ", ".join(payload.get("participants", [])),
        "expenses": payload.get("expenses_raw", ""),
    }
    return render_template(
        "tools/tool_page.html",
        tool=tool,
        view_mode=True,
        saved_output=saved_output,
        saved_input=json.dumps(saved_input),
        share_url=url_for("tools.expense_splitter_view", token=token),
        token=token,
        callout_message="We moved this into Trip Planner. Use Copy to new to switch.",
    )


@tools_bp.route("/trip/<token>", methods=["GET"])
def trip_share_view(token: str):
    tool = get_tool_by_slug("trip-planner")
    if not tool:
        abort(404)
    event = SharedExpenseEvent.query.filter_by(token=token).first()
    now = datetime.now(timezone.utc)
    if not event or not event.expires_at or event.expires_at < now:
        return render_template(
            "tools/trip_share.html",
            error_message="This trip is unavailable or expired.",
        ), 404
    payload = event.to_payload()
    p_type = payload.get("type")
    if p_type != "trip-planner":
        # Fallback for legacy expense payloads
        return render_template(
            "tools/trip_share.html",
            legacy_output=payload.get("settlement_text", ""),
            trip_name=payload.get("event_name", "Shared expenses"),
            error_message=None,
        )
    return render_template(
        "tools/trip_share.html",
        trip_name=payload.get("trip_name", ""),
        currency=payload.get("currency", ""),
        notes=payload.get("notes", ""),
        people=payload.get("people", []),
        budgets=payload.get("budget_summary", []),
        expenses=payload.get("expenses_paid", []),
        planned=payload.get("items_planned", []),
        per_person=payload.get("per_person", []),
        settlements=payload.get("settlement_transfers", []),
        share_url=url_for("tools.trip_share_view", token=token),
        token=token,
        error_message=None,
    )


@tools_bp.route("/grocery/<token>", methods=["GET"])
def grocery_share_view(token: str):
    tool = get_tool_by_slug("grocery-list")
    if not tool:
        abort(404)
    # Render the regular tool page; the client UI loads the shared payload via /api/tools/run
    return render_template(
        "tools/tool_page.html",
        tool=tool,
        view_mode=False,
        share_url=_grocery_share_url(token),
        callout_message="This list is shared. Changes auto-save for anyone with the link.",
    )


@tools_bp.route("/grocery/share/<token>", methods=["GET"])
def grocery_share_redirect(token: str):
    return redirect(_grocery_share_url(token))


@tools_bp.route("/countdown/<token>", methods=["GET"])
def countdown_share_view(token: str):
    tool = get_tool_by_slug("countdown")
    if not tool:
        abort(404)
    return render_template(
        "tools/tool_page.html",
        tool=tool,
        view_mode=False,
        share_url=url_for("tools.countdown_share_view", token=token),
        callout_message="Shared countdown. Save it locally if you want to track it on this device.",
    )


@tools_bp.route("/<slug>", methods=["GET"])
def tool_page(slug: str):
    tool = get_tool_by_slug(slug)
    if not tool or not tool.get("is_enabled"):
        abort(404)
    return render_template("tools/tool_page.html", tool=tool, view_mode=False)


@api_tools_bp.route("/run", methods=["POST"])
def run_tool():
    request_id = str(uuid4())
    # Rate limiting placeholder: integrate limiter middleware before enabling public traffic.

    if not request.is_json:
        return _error_response("INVALID_JSON", "Request body must be JSON.", request_id)

    payload = request.get_json(silent=True)
    if payload is None:
        return _error_response("INVALID_JSON", "Request body must be valid JSON.", request_id)
    if not isinstance(payload, dict):
        return _error_response("INVALID_REQUEST", "Payload must be a JSON object.", request_id)

    tool_slug = payload.get("tool")
    tool_input = payload.get("input")

    if not tool_slug or not isinstance(tool_slug, str):
        return _error_response("INVALID_REQUEST", '"tool" is required and must be a string.', request_id)

    if tool_input is None or not isinstance(tool_input, dict):
        return _error_response("INVALID_REQUEST", '"input" is required and must be an object.', request_id)

    tool = get_tool_by_slug(tool_slug)
    if not tool or not tool.get("is_enabled"):
        return _error_response("TOOL_NOT_AVAILABLE", "Tool not available yet.", request_id)

    try:
        validated_input = _validate_input_against_schema(tool, tool_input, request_id)
    except ValueError as exc:
        return _error_response("INVALID_REQUEST", str(exc), request_id)

    if tool_slug == "resume-bullets":
        try:
            result = run_resume_bullets(validated_input)
            return _build_response(True, data=result, error=None, request_id=request_id), 200
        except Exception as exc:
            return _error_response("TOOL_EXECUTION_FAILED", f"Resume helper failed: {exc}", request_id, status=500)

    if tool_slug == "trip-planner":
        try:
            # Preserve structured fields beyond the schema-defined ones
            merged_input: Dict[str, Any] = dict(validated_input)
            for k, v in (tool_input or {}).items():
                if k not in merged_input:
                    merged_input[k] = v
            if not _free_share_allowed():
                return _share_limit_response(request_id)
            result = run_trip_planner(merged_input)
            expires_at = datetime.fromisoformat(result["expires_at"])
            share_url = url_for("tools.trip_share_view", token=result["token"])
            event = SharedExpenseEvent(
                token=result["token"],
                expires_at=expires_at,
                payload_json=result["payload_json"],
                user_id=_current_user_id(),
            )
            db.session.add(event)
            db.session.commit()
            _record_free_share(result["token"])
            data = {
                "output": result["output"],
                "token": result["token"],
                "share_url": share_url,
                "structured": result.get("structured"),
            }
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("INVALID_INPUT", str(exc), request_id, status=400)
        except Exception as exc:
            db.session.rollback()
            return _error_response("TOOL_EXECUTION_FAILED", f"Trip planner failed: {exc}", request_id, status=500)

    if tool_slug == "daily-phrase":
        try:
            result = run_daily_phrase(validated_input)
            phrase = (result.get("phrase") or "").strip()
            translation = (result.get("translation") or "").strip()
            example = (result.get("example") or "").strip()
            date = (result.get("date") or datetime.now().strftime("%Y-%m-%d")).strip()
            output_lines = [
                f"Date: {date}",
                f"Phrase: {phrase}",
                f"Translation: {translation}",
                f"Example: {example}",
                "Same phrase for everyone today.",
            ]
            data = {
                "output": "\n".join([line for line in output_lines if line]),
                "phrase": phrase,
                "translation": translation,
                "example": example,
                "date": date,
            }
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("INVALID_REQUEST", str(exc), request_id, status=400)
        except Exception as exc:
            return _error_response("TOOL_EXECUTION_FAILED", f"Daily phrase failed: {exc}", request_id, status=500)

    if tool_slug == "decision-helper":
        try:
            result = run_decision_helper(validated_input)
            data = {"output": result.get("output", "")}
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("INVALID_REQUEST", str(exc), request_id, status=400)
        except Exception as exc:
            return _error_response("TOOL_EXECUTION_FAILED", f"Decision helper failed: {exc}", request_id, status=500)

    if tool_slug == "worth-it":
        try:
            result = run_worth_it(validated_input)
            data = {"output": result.get("output", "")}
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("TOOL_INPUT_INVALID", str(exc), request_id, status=400)
        except Exception as exc:
            return _error_response("TOOL_EXECUTION_FAILED", f"Is This Worth It failed: {exc}", request_id, status=500)

    if tool_slug == "social-post-polisher":
        try:
            result = run_social_post_polisher(validated_input)
            data = {"output": result.get("output", "")}
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("TOOL_INPUT_INVALID", str(exc), request_id, status=400)
        except Exception as exc:
            return _error_response("TOOL_EXECUTION_FAILED", f"Social Post Polisher failed: {exc}", request_id, status=500)

    if tool_slug == "grocery-list":
        try:
            # Preserve structured fields beyond the schema-defined ones (token, action, items)
            merged_input: Dict[str, Any] = dict(validated_input)
            for k, v in (tool_input or {}).items():
                if k not in merged_input:
                    merged_input[k] = v

            # Rate limiting hook: integrate limiter middleware before enabling public traffic.
            # Grocery lists can be chatty due to autosave; keep this as a future integration point.

            action = (merged_input.get("action") or "create").strip().lower()
            if action not in {"create", "get", "update"}:
                return _error_response("TOOL_INPUT_INVALID", "Action must be create, get, or update.", request_id, status=400)

            # Lightweight payload size guard (allows larger than MAX_INPUT_CHARS for lists)
            if _aggregate_string_length(merged_input) > 60_000:
                return _error_response("TOOL_INPUT_INVALID", "Payload too large. Try removing some items.", request_id, status=400)

            if action == "create":
                # Create a new shared token record
                if not _free_share_allowed():
                    return _share_limit_response(request_id)
                token = secrets.token_urlsafe(24)[:32]
                created = run_grocery_list_create(merged_input, token=token)
                payload = created.get("payload") or {}
                event = SharedExpenseEvent(
                    token=token,
                    payload_json=json.dumps(payload),
                    user_id=_current_user_id(),
                )
                db.session.add(event)
                db.session.commit()
                _record_free_share(token)
                share_url = _grocery_share_url(token)
                data = {"output": {"token": token, "share_url": share_url, "payload": payload}}
                return _build_response(True, data=data, error=None, request_id=request_id), 200

            # get/update requires token
            token = (merged_input.get("token") or "").strip()
            if not token:
                return _error_response("TOOL_INPUT_INVALID", "Token is required.", request_id, status=400)

            event = SharedExpenseEvent.query.filter_by(token=token).first()
            now = datetime.now(timezone.utc)
            if not event or not event.expires_at or event.expires_at < now:
                return _error_response("NOT_FOUND", "This grocery list link is unavailable or expired.", request_id, status=404)

            current_payload = event.to_payload()
            if current_payload.get("type") != "grocery-list":
                return _error_response("INVALID_REQUEST", "This token does not belong to a grocery list.", request_id, status=400)

            if action == "get":
                run_grocery_list_get(merged_input)
                share_url = _grocery_share_url(token)
                data = {"output": {"token": token, "share_url": share_url, "payload": current_payload}}
                return _build_response(True, data=data, error=None, request_id=request_id), 200

            # update
            updated = run_grocery_list_update(merged_input, current_payload)
            next_payload = updated.get("payload") or {}
            event.payload_json = json.dumps(next_payload)
            db.session.commit()
            share_url = _grocery_share_url(token)
            data = {"output": {"token": token, "share_url": share_url, "payload": next_payload}}
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("TOOL_INPUT_INVALID", str(exc), request_id, status=400)
        except Exception as exc:
            db.session.rollback()
            return _error_response("TOOL_EXECUTION_FAILED", f"Grocery list failed: {exc}", request_id, status=500)

    if tool_slug == "countdown":
        try:
            merged_input: Dict[str, Any] = dict(validated_input)
            for k, v in (tool_input or {}).items():
                if k not in merged_input:
                    merged_input[k] = v

            # Rate limiting hook: integrate limiter middleware before enabling public traffic.
            action = (merged_input.get("action") or "create_share").strip().lower()
            if action not in {"create_share", "get"}:
                return _error_response("TOOL_INPUT_INVALID", "Action must be create_share or get.", request_id, status=400)

            if _aggregate_string_length(merged_input) > 10_000:
                return _error_response("TOOL_INPUT_INVALID", "Payload too large.", request_id, status=400)

            if action == "create_share":
                if not _free_share_allowed():
                    return _share_limit_response(request_id)
                token = secrets.token_urlsafe(24)[:32]
                try:
                    created = run_countdown_create_share(merged_input, token=token)
                    payload = created.get("payload") or {}
                    # Keep countdowns around longer than trip shares
                    event = SharedExpenseEvent(
                        token=token,
                        expires_at=default_expiry(days=365),
                        payload_json=json.dumps(payload),
                        user_id=_current_user_id(),
                    )
                    db.session.add(event)
                    db.session.commit()
                    _record_free_share(token)
                    share_url = url_for("tools.countdown_share_view", token=token)
                    data = {"output": {"token": token, "share_url": share_url, "payload": payload}}
                    return _build_response(True, data=data, error=None, request_id=request_id), 200
                except ValueError:
                    db.session.rollback()
                    return _error_response("invalid_input", "Check the date and try again.", request_id, status=400)
                except Exception as exc:
                    db.session.rollback()
                    ref = _ref_code()
                    logging.exception("countdown share_link_failed ref=%s", ref)
                    friendly = "Something on our side failed while saving your share link. Please refresh and try again."
                    return (
                        jsonify({"ok": False, "error": "share_link_failed", "message": friendly, "ref": ref}),
                        500,
                    )

            token = (merged_input.get("token") or "").strip()
            if not token:
                return _error_response("TOOL_INPUT_INVALID", "Token is required.", request_id, status=400)
            run_countdown_get(merged_input)

            event = SharedExpenseEvent.query.filter_by(token=token).first()
            now = datetime.now(timezone.utc)
            if not event or not event.expires_at or event.expires_at < now:
                return _error_response("NOT_FOUND", "This countdown link is unavailable or expired.", request_id, status=404)
            current_payload = event.to_payload()
            if current_payload.get("type") != "countdown":
                return _error_response("INVALID_REQUEST", "This token does not belong to a countdown.", request_id, status=400)

            share_url = url_for("tools.countdown_share_view", token=token)
            data = {"output": {"token": token, "share_url": share_url, "payload": current_payload}}
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("TOOL_INPUT_INVALID", str(exc), request_id, status=400)
        except Exception as exc:
            db.session.rollback()
            return _error_response("TOOL_EXECUTION_FAILED", f"Countdown failed: {exc}", request_id, status=500)

    if tool_slug == "expense-splitter":
        try:
            result = run_expense_splitter(validated_input)
            expires_at = datetime.fromisoformat(result["expires_at"])
            share_url = url_for("tools.expense_splitter_view", token=result["token"])
            if not _free_share_allowed():
                return _share_limit_response(request_id)
            event = SharedExpenseEvent(
                token=result["token"],
                expires_at=expires_at,
                payload_json=result["payload_json"],
                user_id=_current_user_id(),
            )
            db.session.add(event)
            db.session.commit()
            _record_free_share(result["token"])
            data = {"output": result["output"], "token": result["token"], "share_url": share_url}
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("INVALID_INPUT", str(exc), request_id, status=400)
        except Exception as exc:
            db.session.rollback()
            return _error_response("TOOL_EXECUTION_FAILED", f"Expense splitter failed: {exc}", request_id, status=500)

    return _error_response("TOOL_NOT_AVAILABLE", "Tool not available yet.", request_id)


@api_tools_bp.route("/trip/<token>", methods=["GET"])
def get_trip_payload(token: str):
    request_id = str(uuid4())
    event = SharedExpenseEvent.query.filter_by(token=token).first()
    now = datetime.now(timezone.utc)
    if not event or not event.expires_at or event.expires_at < now:
        return _error_response("NOT_FOUND", "Trip not found or expired.", request_id, status=404)
    payload = event.to_payload()
    if payload.get("type") != "trip-planner":
        return _error_response("INVALID_REQUEST", "Not a trip planner event.", request_id, status=400)
    return _build_response(True, data=payload, error=None, request_id=request_id), 200
