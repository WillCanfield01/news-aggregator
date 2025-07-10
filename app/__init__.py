import os
from flask import Flask, jsonify
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
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
    }

    # Extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "home"
    login_manager.unauthorized_handler = lambda: (jsonify({"error": "Unauthorized"}), 401)

    CORS(app, supports_credentials=True)

    # Blueprints
    from app.routes import auth_routes, news_routes, local_routes, user_routes
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(news_routes.bp)
    app.register_blueprint(local_routes.bp)
    app.register_blueprint(user_routes.bp)

    # --- START THREADS HERE ---
    from app.utils.feed_utils import preload_articles_batched, RSS_FEED_BATCHES
    from app.utils.tasks import start_periodic_refresh, start_periodic_local_refresh

    # Local articles cache can be a global dict, but must persist for the life of the app
    # You may want to move this to a better location or make it a property of app
    local_articles_cache = {}

    with app.app_context():
        # Main news batching
        preload_articles_batched(RSS_FEED_BATCHES[0], use_ai=False)
        start_periodic_refresh(app)
        # Local news refresh
        start_periodic_local_refresh(app, local_articles_cache=local_articles_cache)

    # Optionally, attach the local cache to the app object for use in routes
    app.local_articles_cache = local_articles_cache

    return app