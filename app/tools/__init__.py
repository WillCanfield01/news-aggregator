from __future__ import annotations

from typing import Any
from uuid import uuid4

from flask import Blueprint, jsonify, render_template, request

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

# Placeholder allowlist for future tools
ALLOWED_TOOLS: dict[str, Any] = {}

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


@tools_bp.route("/", methods=["GET"])
@tools_bp.route("", methods=["GET"])
def tools_home():
    return render_template("tools/index.html")


@tools_bp.route("/<slug>", methods=["GET"])
def tool_page(slug: str):
    title_words = (slug or "").replace("-", " ").replace("_", " ").split()
    display_title = " ".join(word.capitalize() for word in title_words) or "Tool"
    return render_template("tools/tool_page.html", slug=slug, tool_title=display_title)


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

    total_chars = _aggregate_string_length(tool_input)
    if total_chars > MAX_INPUT_CHARS:
        return _error_response(
            "INPUT_TOO_LARGE",
            f"Input is too large. Limit is {MAX_INPUT_CHARS} characters across text fields.",
            request_id,
        )

    if tool_slug not in ALLOWED_TOOLS:
        return _error_response("TOOL_NOT_AVAILABLE", "Tool not available yet.", request_id)

    # Placeholder response path for future tool execution logic.
    return _build_response(True, data={"message": "Tool execution stub."}, error=None, request_id=request_id), 200
