# patchpal/storage.py
import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///patchpal.db")

# Normalize to psycopg v3 driver no matter what the platform gives us
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg2://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL)
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

Base.metadata.create_all(engine)
