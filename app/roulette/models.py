from datetime import datetime, date
from app.extensions import db

class TimelineRound(db.Model):
    __tablename__ = "timeline_rounds"
    __table_args__ = {"extend_existing": True}  # <-- important

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    round_date = db.Column(db.Date, unique=True, index=True, nullable=False)

    real_title = db.Column(db.String(300), nullable=False)
    real_source_url = db.Column(db.String(600), nullable=False)
    fake1_title = db.Column(db.String(300), nullable=False)
    fake2_title = db.Column(db.String(300), nullable=False)

    # Unsplash thumbnails + tiny attribution
    real_img_url  = db.Column(db.String(600), nullable=True)
    real_img_attr = db.Column(db.String(300), nullable=True)
    fake1_img_url  = db.Column(db.String(600), nullable=True)
    fake1_img_attr = db.Column(db.String(300), nullable=True)
    fake2_img_url  = db.Column(db.String(600), nullable=True)
    fake2_img_attr = db.Column(db.String(300), nullable=True)

    correct_index = db.Column(db.Integer, nullable=False, default=0)
    published_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class TimelineGuess(db.Model):
    __tablename__ = "timeline_guesses"
    __table_args__ = {"extend_existing": True}  # <-- important

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    round_id = db.Column(db.Integer, db.ForeignKey("timeline_rounds.id"), index=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    choice_index = db.Column(db.Integer, nullable=False)
    is_correct = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip_hash = db.Column(db.String(64), index=True)


class TimelineStreak(db.Model):
    __tablename__ = "timeline_streaks"
    __table_args__ = {"extend_existing": True}  # <-- important

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, unique=True, index=True, nullable=True)
    current_streak = db.Column(db.Integer, default=0, nullable=False)
    last_play_date = db.Column(db.Date, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
