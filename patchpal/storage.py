# patchpal/storage.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///patchpal.db")

# Force psycopg v3
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=180,
    pool_size=5,
    max_overflow=5,
    connect_args={"sslmode": "require"} if DATABASE_URL.startswith("postgresql+psycopg://") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True)
    team_name = Column(String)
    tz = Column(String, default="America/Boise")
    post_channel = Column(String, nullable=True)
    post_time = Column(String, default="09:00")
    tone = Column(String, default="simple")

    # 🔒 Billing/trial
    plan = Column(String, default="trial")             # 'trial' | 'pro' | 'canceled'
    trial_ends_at = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    subscription_id = Column(String, nullable=True)
    contact_user_id = Column(String, nullable=True)    # Slack user to DM
    last_billing_nag = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class PostLog(Base):
    __tablename__ = "post_logs"
    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True)
    post_date = Column(String, index=True)

Base.metadata.create_all(engine)

# Tiny migrations for existing DBs
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
        if "contact_user_id" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN contact_user_id VARCHAR NULL"))
        if "last_billing_nag" not in cols:
            conn.execute(text("ALTER TABLE workspaces ADD COLUMN last_billing_nag TIMESTAMP NULL"))
except Exception:
    pass
