from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from flask import Blueprint, abort, jsonify, render_template, request, url_for

from app.extensions import db
from app.tools.handlers import run_expense_splitter, run_resume_bullets, run_trip_planner, run_daily_phrase
from app.tools.models import SharedExpenseEvent
from app.tools.registry import get_enabled_tools, get_tool_by_slug

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
            if required:
                raise ValueError(f'"{name}" is required.')
            incoming = default_value or ""

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
    return render_template("tools/index.html", tools=enabled_tools)


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
            result = run_trip_planner(merged_input)
            expires_at = datetime.fromisoformat(result["expires_at"])
            share_url = url_for("tools.trip_share_view", token=result["token"])
            event = SharedExpenseEvent(
                token=result["token"],
                expires_at=expires_at,
                payload_json=result["payload_json"],
            )
            db.session.add(event)
            db.session.commit()
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
            output_lines = [
                f"Date: {result.get('date')}",
                f"Phrase: {result.get('phrase')}",
                f"Translation: {result.get('translation')}",
                f"Example: {result.get('example')}",
                "Same phrase for everyone today.",
            ]
            data = {
                "output": "\n".join([line for line in output_lines if line]),
                "phrase": result.get("phrase"),
                "translation": result.get("translation"),
                "example": result.get("example"),
                "date": result.get("date"),
            }
            return _build_response(True, data=data, error=None, request_id=request_id), 200
        except ValueError as exc:
            return _error_response("INVALID_REQUEST", str(exc), request_id, status=400)
        except Exception as exc:
            return _error_response("TOOL_EXECUTION_FAILED", f"Daily phrase failed: {exc}", request_id, status=500)

    if tool_slug == "expense-splitter":
        try:
            result = run_expense_splitter(validated_input)
            expires_at = datetime.fromisoformat(result["expires_at"])
            share_url = url_for("tools.expense_splitter_view", token=result["token"])
            event = SharedExpenseEvent(
                token=result["token"],
                expires_at=expires_at,
                payload_json=result["payload_json"],
            )
            db.session.add(event)
            db.session.commit()
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
