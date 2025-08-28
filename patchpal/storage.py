# patchpal/storage.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    text,
    inspect,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# DB URL normalization (prefer psycopg v3 driver on Postgres)
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///patchpal.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    if not (
        DATABASE_URL.startswith("postgresql+psycopg://")
        or DATABASE_URL.startswith("postgresql+psycopg2://")
    ):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

# SQLite needs this arg; Postgres does not.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True, unique=True, nullable=False)
    team_name = Column(String, nullable=True)

    # posting prefs
    tz = Column(String, default="America/Boise")
    post_channel = Column(String, nullable=True)
    post_time = Column(String, default="09:00")
    tone = Column(String, default="simple")

    # relevance controls
    stack_mode = Column(String, default="universal")  # 'universal' | 'stack'
    stack_tokens = Column(Text, nullable=True)        # csv e.g. "windows,ms365,chrome"
    ignore_tokens = Column(Text, nullable=True)       # reserved for feedback buttons

    # billing / trial
    plan = Column(String, default="trial")            # 'trial' | 'pro' | 'canceled'
    trial_ends_at = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    subscription_id = Column(String, nullable=True)
    customer_id = Column(String, nullable=True)       # for Stripe Portal

    # contacts / nags
    contact_user_id = Column(String, nullable=True)
    last_billing_nag = Column(DateTime, nullable=True)
    last_trial_warn = Column(DateTime, nullable=True)
    last_payment_fail_nag = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Installation(Base):
    __tablename__ = "installations"
    team_id = Column(String(64), primary_key=True)
    team_name = Column(String(255))
    bot_token = Column(Text, nullable=False)
    bot_user  = Column(String(64))
    scopes    = Column(Text)
    installed_by_user_id = Column(String(64))
    enterprise_id = Column(String(64))
    installed_at = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PostLog(Base):
    __tablename__ = "post_logs"
    __table_args__ = (
        UniqueConstraint("team_id", "post_date", name="uq_postlog_team_date"),
    )

    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True, nullable=False)
    post_date = Column(String, index=True, nullable=False)  # YYYY-MM-DD
    created_at = Column(DateTime, default=datetime.utcnow)

class RecentPost(Base):
    __tablename__ = "recent_posts"
    team_id   = Column(String, primary_key=True)
    item_key  = Column(String, primary_key=True)
    posted_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

# --- helpers ---

def _dialect_name(bind) -> str:
    try:
        return bind.dialect.name
    except Exception:
        return ""

def recent_keys(team_id: str, remember_days: int) -> set[str]:
    """Return keys posted within the last `remember_days`, cross-dialect safe."""
    if not team_id:
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=remember_days)
    with SessionLocal() as db:
        rows = (
            db.query(RecentPost.item_key)
              .filter(RecentPost.team_id == team_id, RecentPost.posted_at >= cutoff)
              .all()
        )
    return {r[0] for r in rows}

def remember_posts(team_id: str, item_keys: list[str]) -> None:
    """Upsert recent posts; works on Postgres and SQLite; merges otherwise."""
    if not team_id or not item_keys:
        return
    with SessionLocal() as db:
        bind = db.get_bind()
        dname = _dialect_name(bind)
        values = [{"team_id": team_id, "item_key": k} for k in item_keys]

        try:
            if dname.startswith("postgres"):
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                stmt = pg_insert(RecentPost).values(values).on_conflict_do_update(
                    index_elements=[RecentPost.team_id, RecentPost.item_key],
                    set_={"posted_at": func.now()},
                )
                db.execute(stmt)
            elif dname == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert
                stmt = sqlite_insert(RecentPost).values(values).on_conflict_do_update(
                    index_elements=[RecentPost.team_id, RecentPost.item_key],
                    set_={"posted_at": func.now()},
                )
                db.execute(stmt)
            else:
                # Portable fallback
                now = datetime.now(timezone.utc)
                for k in item_keys:
                    db.merge(RecentPost(team_id=team_id, item_key=k, posted_at=now))
        finally:
            db.commit()

def delete_recent_for_team(team_id: str) -> None:
    with SessionLocal() as db:
        db.query(RecentPost).filter_by(team_id=team_id).delete(synchronize_session=False)
        db.commit()

# ---------------------------------------------------------------------------
# Create tables (do this here; remove the duplicate call in app.py)
# ---------------------------------------------------------------------------
Base.metadata.create_all(engine)

# ---------------------------------------------------------------------------
# Tiny migrations (idempotent)
# ---------------------------------------------------------------------------
try:
    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("workspaces")}

    with engine.begin() as conn:
        # legacy columns (safe if already present)
        if "tone" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN tone VARCHAR"))
        if "plan" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN plan VARCHAR DEFAULT 'trial'"))
        if "trial_ends_at" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN trial_ends_at TIMESTAMP NULL"))
        if "paid_at" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN paid_at TIMESTAMP NULL"))
        if "subscription_id" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN subscription_id VARCHAR NULL"))
        if "customer_id" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN customer_id VARCHAR NULL"))
        if "contact_user_id" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN contact_user_id VARCHAR NULL"))
        if "last_billing_nag" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN last_billing_nag TIMESTAMP NULL"))
        if "last_trial_warn" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN last_trial_warn TIMESTAMP NULL"))
        if "last_payment_fail_nag" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN last_payment_fail_nag TIMESTAMP NULL"))

        # NEW relevance fields
        if "stack_mode" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN stack_mode VARCHAR DEFAULT 'universal'"))
            conn.execute(text("UPDATE workspaces SET stack_mode='universal' WHERE stack_mode IS NULL"))
        if "stack_tokens" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN stack_tokens TEXT NULL"))
        if "ignore_tokens" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN ignore_tokens TEXT NULL"))

        # Ensure uniqueness and helpful indexes
        conn.execute(
            text("CREATE UNIQUE INDEX IF NOT EXISTS uq_postlog_team_date ON post_logs (team_id, post_date)")
        )
        conn.execute(
            text("CREATE UNIQUE INDEX IF NOT EXISTS uq_workspaces_team_id ON workspaces (team_id)")
        )

except Exception:
    # don't block startup if ALTER/INDEX creation fails (first boot / transient states)
    pass
