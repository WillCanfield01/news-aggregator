# config.py

import os
from sqlalchemy.pool import QueuePool

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-dev-key")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "pool_size": 5,
        "max_overflow": 10,
        "poolclass": QueuePool
    }

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DEV_DATABASE_URL",
        "sqlite:///instance/local.db"  # keep relative and portable
    )

class ProductionConfig(Config):
    DEBUG = False

    @classmethod
    def get_database_uri(cls):
        uri = os.getenv("DATABASE_URL", "")
        if uri.startswith("postgres://"):
            uri = uri.replace("postgres://", "postgresql://", 1)
        if uri and "sslmode" not in uri:
            uri += "?sslmode=require"
        return uri

    SQLALCHEMY_DATABASE_URI = get_database_uri.__func__()

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}