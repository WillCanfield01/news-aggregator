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
