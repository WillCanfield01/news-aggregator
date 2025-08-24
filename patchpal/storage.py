from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///patchpal.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True)
    team_name = Column(String)
    tz = Column(String, default="America/Boise")
    post_channel = Column(String, nullable=True)   # channel id
    post_time = Column(String, default="09:00")    # HH:MM 24h
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class PostLog(Base):
    __tablename__ = "post_logs"
    id = Column(Integer, primary_key=True)
    team_id = Column(String, index=True)
    post_date = Column(String, index=True)         # YYYY-MM-DD UTC

Base.metadata.create_all(engine)