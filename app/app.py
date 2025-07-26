import os
from datetime import datetime
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    # ---- Configs ----
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///local.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.secret_key = os.environ.get("SECRET_KEY", "super-secret-dev-key")
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax'
    )

    # ---- Initialize Extensions ----
    db.init_app(app)
    login_manager.init_app(app)  # <-- THIS LINE IS MISSING!
    login_manager.login_view = "aggregator.login"  # Optional: set login page

    from app.aggregator import aggregator_bp, start_background_tasks
    from app.reddit_articles import bp as reddit_bp

    app.register_blueprint(aggregator_bp)
    app.register_blueprint(reddit_bp)

    # --- LANDING PAGE ROUTE ---
    @app.route("/")
    def landing():
        return render_template("index.html", year=datetime.now().year)

    with app.app_context():
        start_background_tasks()

    return app

if __name__ == "__main__":
    app = create_app()
    # with app.app_context():
    #     db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
