import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from config import config

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_name='default'):
    app = Flask(__name__)

    # Load and sanitize database URL
    uri = os.environ.get("DATABASE_URL", "")
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql://", 1)
    if uri and "sslmode" not in uri:
        uri += "?sslmode=require"

    app.config["SQLALCHEMY_DATABASE_URI"] = uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.secret_key = os.getenv("SECRET_KEY", "super-secret-dev-key")

    db.init_app(app)
    login_manager.init_app(app)

    # Register blueprints
    from app.routes import auth_routes, news_routes, local_routes, user_routes
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(news_routes.bp)
    app.register_blueprint(local_routes.bp)
    app.register_blueprint(user_routes.bp)

    return app
