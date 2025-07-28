import os
from datetime import datetime
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

db = SQLAlchemy()
login_manager = LoginManager()

def schedule_daily_reddit_article(app):
    def scheduled_job():
        with app.app_context():
            from app.reddit_articles import generate_article_for_today
            generate_article_for_today()
            print("✅ Daily Reddit article generated at", datetime.now())

    scheduler = BackgroundScheduler(timezone=pytz.timezone("America/Denver"))  # MST/MDT
    # Runs every day at 17:00 (5 PM) Mountain Time
    scheduler.add_job(scheduled_job, "cron", hour=17, minute=0)
    scheduler.start()

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
    login_manager.init_app(app)
    login_manager.login_view = "aggregator.login"

    from app.aggregator import aggregator_bp, start_background_tasks
    from app.reddit_articles import bp as reddit_bp

    app.register_blueprint(aggregator_bp)
    app.register_blueprint(reddit_bp)

    @app.route("/")
    def landing():
        return render_template("index.html", year=datetime.now().year)

    # === ADD THIS FOR SITEMAP ===
    @app.route('/sitemap.xml')
    def sitemap():
        base_url = "https://therealroundup.com"  # Make sure this is your real domain
        articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
        urls = [f"{base_url}/reddit-articles/articles/{a.filename}" for a in articles]

        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
        xml.append(f"""
        <url>
            <loc>{base_url}/</loc>
            <priority>1.0</priority>
        </url>
        """)
        for url in urls:
            xml.append(f"""
        <url>
            <loc>{url}</loc>
            <priority>0.8</priority>
        </url>
        """)
        xml.append('</urlset>')
        sitemap_xml = "\n".join(xml)
        return Response(sitemap_xml, mimetype='application/xml')

    with app.app_context():
        start_background_tasks()
        schedule_daily_reddit_article(app)

    return app


if __name__ == "__main__":
    app = create_app()
    # with app.app_context():
    #     db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
