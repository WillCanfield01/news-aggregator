# app/escape/models.py
# -*- coding: utf-8 -*-
"""
Mini Escape Rooms - Data Models

Tables:
- EscapeRoom: one row per daily room (date_key-unique) storing validated JSON and difficulty
- EscapeAttempt: one row per player's run (anonymous-friendly), for leaderboards & analytics

Notes:
- Room JSON can be either the legacy flat shape (top-level puzzles[]) or the new
  Trailroom shape (trail.rooms[].routes[].puzzle + final). We store it as a single blob.
- Keep models lean: business logic stays in core.py and routes.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime

from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB

from app.extensions import db


class EscapeRoom(db.Model):
    __tablename__ = "escape_rooms"

    id = db.Column(db.Integer, primary_key=True)

    # ISO date string, e.g. "2025-09-01"
    date_key = db.Column(db.String(16), unique=True, index=True, nullable=False)

    # Validated room JSON (flat or trail). JSONB on Postgres, JSON on SQLite.
    json_blob = db.Column(JSONB().with_variant(db.JSON, "sqlite"), nullable=False)

    difficulty = db.Column(db.String(16), index=True, default="medium", nullable=False)

    # Used by "recent_rooms" queries; make it indexed for speed.
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Optional: increments when you force-regenerate a day
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

    # If/when you add auth, you can link users. Keep nullable to allow anonymous play.
    user_id = db.Column(db.Integer, index=True, nullable=True)

    # ISO date of the room this attempt belongs to
    date_key = db.Column(db.String(16), index=True, nullable=False)

    started_at = db.Column(db.DateTime, nullable=False)
    finished_at = db.Column(db.DateTime, nullable=True)

    # Milliseconds to finish; null if not finished or unsuccessful
    time_ms = db.Column(db.Integer, index=True, nullable=True)

    # Whether the player escaped successfully
    success = db.Column(db.Boolean, index=True, default=False, nullable=False)

    # Extra metadata (UA hash, coarse region, device hints, client version, etc.)
    meta = db.Column(JSONB().with_variant(db.JSON, "sqlite"), nullable=False, default=dict)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Leaderboard accelerator: (date_key, success, time_ms ASC)
    __table_args__ = (
        Index("ix_attempts_daily_lb", "date_key", "success", "time_ms"),
    )

    def __repr__(self) -> str:
        t = self.time_ms if self.time_ms is not None else "-"
        return f"<EscapeAttempt id={self.id} date={self.date_key} success={self.success} time_ms={t}>"

    @property
    def duration_seconds(self) -> Optional[float]:
        return None if self.time_ms is None else self.time_ms / 1000.0

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


# ---------- Convenience query helpers (not tables) ----------

class DailyLeaderboardView:
    """
    Lightweight helper to fetch today's/top leaderboards.
    Use from routes without polluting them with query details.
    """

    @staticmethod
    def top_for_day(date_key: str, limit: int = 50):
        q = (
            db.session.query(EscapeAttempt)
            .filter(
                EscapeAttempt.date_key == date_key,
                EscapeAttempt.success.is_(True),
                EscapeAttempt.time_ms.isnot(None),
            )
        )
        rows = q.all()
        # Sort by adjusted time ASC, then chips_remaining DESC
        rows.sort(
            key=lambda a: (
                a.time_ms if a.time_ms is not None else 10 ** 12,
                -int((a.meta or {}).get("chips_remaining", 0)),
            )
        )
        out = []
        for idx, row in enumerate(rows[:limit], start=1):
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
