# patchpal/storage.py
from __future__ import annotations
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

# --- DB URL normalization (use psycopg v3) ----------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///patchpal.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    # add +psycopg if no driver specified
    if not (DATABASE_URL.startswith("postgresql+psycopg://") or DATABASE_URL.startswith("postgresql+psycopg2://")):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

# SQLite needs a special arg, Postgres does not
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()

# --- Models -----------------------------------------------------------------
class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True)
    team_name = Column(String, nullable=True)

    # posting prefs
    tz = Column(String, default="America/Boise")
    post_channel = Column(String, nullable=True)
    post_time = Column(String, default="09:00")
    tone = Column(String, default="simple")

    # billing / trial
    plan = Column(String, default="trial")                  # trial | pro | canceled
    trial_ends_at = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    subscription_id = Column(String, nullable=True)
    customer_id = Column(String, nullable=True)             # for Stripe Portal

    # contacts / nags
    contact_user_id = Column(String, nullable=True)
    last_billing_nag = Column(DateTime, nullable=True)
    last_trial_warn = Column(DateTime, nullable=True)
    last_payment_fail_nag = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class PostLog(Base):
    __tablename__ = "post_logs"

    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True, nullable=False)
    post_date = Column(String, index=True, nullable=False)  # YYYY-MM-DD
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Create tables -----------------------------------------------------------
Base.metadata.create_all(engine)

# --- Tiny migrations for existing DBs ---------------------------------------
try:
    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("workspaces")}
    with engine.begin() as conn:
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
except Exception:
    # don't block startup if ALTER fails (first boot on SQLite, etc.)
    pass
