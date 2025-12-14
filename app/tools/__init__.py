from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from flask import Blueprint, abort, jsonify, render_template, request

from app.tools.handlers import run_resume_bullets
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


@tools_bp.route("/<slug>", methods=["GET"])
def tool_page(slug: str):
    tool = get_tool_by_slug(slug)
    if not tool or not tool.get("is_enabled"):
        abort(404)
    return render_template("tools/tool_page.html", tool=tool)


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

    return _error_response("TOOL_NOT_AVAILABLE", "Tool not available yet.", request_id)
