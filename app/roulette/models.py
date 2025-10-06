from datetime import datetime, date
from app.extensions import db

# One row per day
class TimelineRound(db.Model):
    __tablename__ = "timeline_rounds"

    id = db.Column(db.Integer, primary_key=True)
    round_date = db.Column(db.Date, unique=True, index=True, nullable=False)

    # One true event + 2 fakes (short, punchy)
    real_title = db.Column(db.String(300), nullable=False)
    real_source_url = db.Column(db.String(600), nullable=False)

    fake1_title = db.Column(db.String(300), nullable=False)
    fake2_title = db.Column(db.String(300), nullable=False)

    # Index of correct card BEFORE any runtime shuffle (0 = real, 1 = fake1, 2 = fake2)
    correct_index = db.Column(db.Integer, nullable=False, default=0)

    published_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class TimelineGuess(db.Model):
    __tablename__ = "timeline_guesses"

    id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey("timeline_rounds.id"), index=True, nullable=False)

    # Optional user; leave null for guests
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)

    # 0/1/2 which card they chose (in the shuffled UI order)
    choice_index = db.Column(db.Integer, nullable=False)
    is_correct = db.Column(db.Boolean, default=False, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip_hash = db.Column(db.String(64), index=True)  # store a hashed IP or anonymized token


class TimelineStreak(db.Model):
    """
    Server-side streaks for logged-in users.
    Guests still get client-side streak via cookie, but if a user is logged in,
    this keeps an authoritative streak counter.
    """
    __tablename__ = "timeline_streaks"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, index=True)
    current_streak = db.Column(db.Integer, default=0, nullable=False)
    last_play_date = db.Column(db.Date, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
