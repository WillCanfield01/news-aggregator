
# app/escape/models.py (REWRITE)
# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from flask import current_app
from sqlalchemy import func
from app.extensions import db


class EscapeRoom(db.Model):
    __tablename__ = "escape_rooms"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), unique=True, index=True, nullable=False)  # YYYY-MM-DD
    theme = db.Column(db.String(120), nullable=False)
    minigames_json = db.Column(db.JSON, nullable=False)       # public-safe config
    server_private_json = db.Column(db.JSON, nullable=False)  # seeds, etc (never sent to client)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<EscapeRoom {self.date}>"


class EscapeScore(db.Model):
    __tablename__ = "escape_scores"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), index=True, nullable=False)
    nickname = db.Column(db.String(40), nullable=True)  # optional future use
    total_time_ms = db.Column(db.Integer, nullable=False)
    finished_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)
    ip_hash = db.Column(db.String(64), nullable=False)  # to discourage duplicates

    def __repr__(self) -> str:
        return f"<EscapeScore {self.date} {self.total_time_ms}ms>"

    @staticmethod
    def top_for_day(date: str, limit: int = 50) -> List["EscapeScore"]:
        return (EscapeScore.query
                .filter(EscapeScore.date == date)
                .order_by(EscapeScore.total_time_ms.asc(), EscapeScore.finished_at.asc())
                .limit(limit)
                .all())

    @staticmethod
    def personal_bests(ip_hash: str, limit: int = 10) -> List["EscapeScore"]:
        return (EscapeScore.query
                .filter(EscapeScore.ip_hash == ip_hash)
                .order_by(EscapeScore.total_time_ms.asc())
                .limit(limit)
                .all())
