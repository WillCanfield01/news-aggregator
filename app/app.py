import os
from datetime import datetime
from flask import Flask, render_template, Response, url_for, current_app
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

    scheduler = BackgroundScheduler(timezone=pytz.timezone("America/Denver"))
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

    # ==== MOVE THE IMPORT HERE! ====
    from app.models import CommunityArticle

    @app.route("/")
    def landing():
        return render_template("index.html", year=datetime.now().year)

    # === SITEMAP ROUTE ===
    @app.route("/sitemap.xml")
    def sitemap():
        try:
            articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
            base_url = "https://therealroundup.com"
            urlset = [
                f"""<url>
    <loc>{base_url}{url_for('reddit_articles.read_article', filename=a.filename)}</loc>
    <lastmod>{a.date.strftime('%Y-%m-%d') if a.date else ''}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
</url>""" for a in articles
            ]
            # Always include homepage!
            home_url = f"""<url>
    <loc>{base_url}/</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
</url>"""
            sitemap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{home_url}
{''.join(urlset)}
</urlset>"""

            return Response(sitemap_xml, mimetype="application/xml")
        except Exception as e:
            current_app.logger.error(f"Sitemap error: {e}")
            return Response("Internal Server Error", status=500)

    with app.app_context():
        start_background_tasks()
        schedule_daily_reddit_article(app)  # Schedule the daily job

    return app

@app.route("/robots.txt")
def robots():
    return current_app.send_static_file("robots.txt")

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
