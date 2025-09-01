# app/escape/models.py
# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Data Models

Tables:
- EscapeRoom: one row per daily room (date_key-unique) storing validated JSON and difficulty
- EscapeAttempt: one row per player's run (anonymous-friendly), for leaderboards & analytics

Notes:
- Keep models lean: let business logic live in app/escape/core.py and routes.py.
- If you introduce authentication later, you can link user_id to your existing User model.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Index
from app import db
from app.extensions import db   # << change

class EscapeRoom(db.Model):
    __tablename__ = "escape_rooms"

    id = db.Column(db.Integer, primary_key=True)
    # e.g., "2025-09-01" (ISO date). We keep it as string for portability/readability.
    date_key = db.Column(db.String(16), unique=True, index=True, nullable=False)

    # Validated room JSON blob (see core.validate_room); stored as JSONB on Postgres if available.
    json_blob = db.Column(JSONB().with_variant(db.JSON, "sqlite"), nullable=False)

    difficulty = db.Column(db.String(16), index=True, default="medium", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Optional: track versioning/regen counts if you ever force_regen
    regen_count = db.Column(db.Integer, default=0, nullable=False)

    def __repr__(self) -> str:
        return f"<EscapeRoom id={self.id} date_key={self.date_key} diff={self.difficulty}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "date_key": self.date_key,
            "json_blob": self.json_blob,
            "difficulty": self.difficulty,
            "created_at": self.created_at.isoformat() + "Z",
            "regen_count": self.regen_count,
        }


class EscapeAttempt(db.Model):
    __tablename__ = "escape_attempts"

    id = db.Column(db.Integer, primary_key=True)

    # Link to a user if you have auth; keep nullable to support anonymous play.
    user_id = db.Column(db.Integer, index=True, nullable=True)

    # Which daily room this attempt corresponds to (ISO date string)
    date_key = db.Column(db.String(16), index=True, nullable=False)

    started_at = db.Column(db.DateTime, nullable=False)
    finished_at = db.Column(db.DateTime, nullable=True)

    # Milliseconds to finish; null if not finished
    time_ms = db.Column(db.Integer, index=True, nullable=True)

    # Whether the player escaped successfully
    success = db.Column(db.Boolean, index=True, default=False, nullable=False)

    # Extra metadata (UA hashes, coarse region, device hints, client version, etc.)
    meta = db.Column(JSONB().with_variant(db.JSON, "sqlite"), nullable=False, default=dict)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Composite index to accelerate leaderboards: (date_key, success, time_ms ASC)
    __table_args__ = (
        Index("ix_attempts_daily_lb", "date_key", "success", "time_ms"),
    )

    def __repr__(self) -> str:
        t = self.time_ms if self.time_ms is not None else "-"
        return f"<EscapeAttempt id={self.id} date={self.date_key} success={self.success} time_ms={t}>"

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.time_ms is None:
            return None
        return self.time_ms / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "date_key": self.date_key,
            "started_at": self.started_at.isoformat() + "Z",
            "finished_at": self.finished_at.isoformat() + "Z" if self.finished_at else None,
            "time_ms": self.time_ms,
            "success": self.success,
            "meta": self.meta or {},
            "created_at": self.created_at.isoformat() + "Z",
        }


# ---------- Convenience query helpers (optional, not tables) ----------

class DailyLeaderboardView:
    """
    Lightweight helper to fetch today's/top leaderboards.
    Use from routes without polluting them with query details.
    """

    @staticmethod
    def top_for_day(date_key: str, limit: int = 50):
        """
        Return a list of (attempt, rank) for the given day, ordered by best time.
        Success-only, non-null time_ms.
        """
        q = (
            db.session.query(EscapeAttempt)
            .filter(
                EscapeAttempt.date_key == date_key,
                EscapeAttempt.success.is_(True),
                EscapeAttempt.time_ms.isnot(None),
            )
            .order_by(EscapeAttempt.time_ms.asc(), EscapeAttempt.id.asc())
            .limit(limit)
        )
        rows = q.all()
        out = []
        for idx, row in enumerate(rows, start=1):
            out.append({"rank": idx, "attempt": row})
        return out

    @staticmethod
    def personal_bests(user_id: int, limit: int = 20):
        """
        Return the user's best successful times across days.
        """
        q = (
            db.session.query(EscapeAttempt)
            .filter(
                EscapeAttempt.user_id == user_id,
                EscapeAttempt.success.is_(True),
                EscapeAttempt.time_ms.isnot(None),
            )
            .order_by(EscapeAttempt.time_ms.asc(), EscapeAttempt.id.asc())
            .limit(limit)
        )
        return q.all()
