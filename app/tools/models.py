from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.extensions import db


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_expiry(days: int = 14) -> datetime:
    return _utcnow() + timedelta(days=days)


class SharedExpenseEvent(db.Model):
    __tablename__ = "shared_expense_events"

    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), unique=True, index=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow, index=True, nullable=False)
    expires_at = db.Column(db.DateTime(timezone=True), default=default_expiry, index=True, nullable=False)
    payload_json = db.Column(db.Text, nullable=False)

    def to_payload(self):
        try:
            return json.loads(self.payload_json or "{}")
        except Exception:
            return {}


class SharedToolLink(db.Model):
    __tablename__ = "shared_tool_links"

    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(64), unique=True, index=True, nullable=False)
    tool = db.Column(db.String(50), index=True, nullable=False)
    payload_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)

    def to_payload(self):
        try:
            return json.loads(self.payload_json or "{}")
        except Exception:
            return {}


class DailyLanguagePhrase(db.Model):
    __tablename__ = "daily_language_phrases"
    __table_args__ = (db.UniqueConstraint("date", "language", "level", name="uq_daily_phrase_date_lang_level"),)

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, index=True)
    language = db.Column(db.String(50), nullable=False, index=True)
    level = db.Column(db.String(50), nullable=False, index=True)
    phrase = db.Column(db.Text, nullable=False)
    translation = db.Column(db.Text, nullable=False)
    example = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
