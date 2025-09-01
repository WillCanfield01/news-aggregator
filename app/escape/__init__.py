# app/escape/__init__.py
# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Blueprint Factory

This module exposes `create_escape_bp()` which:
- Creates the Flask blueprint for the Escape feature.
- Attaches route handlers from routes.py.
- Keeps template/static resolution tidy without duplicating logic.

Usage (in your app factory):
    from app.escape import create_escape_bp
    app.register_blueprint(create_escape_bp(), url_prefix="/escape")

Optionally start the daily scheduler (once per process):
    from app.escape.core import schedule_daily_generation
    schedule_daily_generation(app)
"""

from __future__ import annotations
from flask import Blueprint


def create_escape_bp() -> Blueprint:
    """
    Create and return the blueprint for the Escape module.
    Templates live in app/templates/escape, and static in app/static/escape.
    """
    # Note: We point to the parent-level template/static folders and rely on
    # the standard app template loader to resolve "escape/*.html".
    bp = Blueprint(
        "escape",
        __name__,
        template_folder="../templates/escape",
        static_folder="../static/escape",
    )

    # Attach routes
    from .routes import init_routes
    init_routes(bp)

    return bp
