# patchpal/storage.py
import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///patchpal.db")

# Force psycopg v3 driver
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

# Render/Postgres can drop idle conns; use pre_ping + recycle; require SSL just in case
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,         # test and refresh dead conns
    pool_recycle=180,           # recycle before typical 5–10 min idles
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class PostLog(Base):
    __tablename__ = "post_logs"
    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True)
    post_date = Column(String, index=True)
