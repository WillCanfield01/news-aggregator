# app/app.py
import os, re
from datetime import datetime
from flask import Flask, render_template, Response, url_for, current_app, jsonify
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
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ---- Configs ----
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///local.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
    app.secret_key = os.environ.get("SECRET_KEY", "super-secret-dev-key")
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

    # ---- Extensions ----
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "aggregator.login"

    # ---- Blueprints ----
    from app.aggregator import aggregator_bp, start_background_tasks
    from app.reddit_articles import bp as reddit_bp
    app.register_blueprint(aggregator_bp)
    app.register_blueprint(reddit_bp)

    from app.models import CommunityArticle

    # ---- Routes ----
    @app.route("/")
    def landing():
        latest = CommunityArticle.query.order_by(CommunityArticle.date.desc()).first()
        return render_template("index.html", year=datetime.now().year, latest=latest)

    @app.route("/sitemap.xml")
    def sitemap():
        try:
            articles = CommunityArticle.query.order_by(CommunityArticle.date.desc()).all()
            urlset = []
            urlset.append(f"""<url>
<loc>{url_for('landing', _external=True)}</loc>
<changefreq>daily</changefreq><priority>1.0</priority>
</url>""")
            for a in articles:
                loc = url_for("all_articles.read_article", filename=a.filename, _external=True)
                lastmod = a.date.strftime("%Y-%m-%d") if a.date else ""
                urlset.append(f"""<url>
<loc>{loc}</loc>
<lastmod>{lastmod}</lastmod>
<changefreq>weekly</changefreq><priority>0.8</priority>
</url>""")
            sitemap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{''.join(urlset)}
</urlset>"""
            return Response(sitemap_xml, content_type="application/xml; charset=utf-8")
        except Exception as e:
            current_app.logger.error(f"Sitemap error: {e}")
            return Response("Internal Server Error", status=500)

    @app.route("/api/reddit-feature")
    def reddit_feature():
        a = CommunityArticle.query.order_by(CommunityArticle.date.desc()).first()
        if not a:
            return jsonify({"title": "", "summary": ""}), 404
        plain = re.sub(r'\!\[.*?\]\(.*?\)', '', a.content)
        plain = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', plain)
        plain = re.sub(r'\*\*|\*|__|_', '', plain)
        words = plain.split()
        summary = " ".join(words[:55]) + ("..." if len(words) > 55 else "")
        return jsonify({
            "title": a.title,
            "summary": summary,
            "url": url_for('all_articles.read_article', filename=a.filename, _external=True)
        })

    @app.route("/robots.txt")
    def robots():
        return (
            "User-agent: *\n"
            "Allow: /\n"
            "Sitemap: https://therealroundup.com/sitemap.xml\n",
            200,
            {"Content-Type": "text/plain"},
        )

    # ---- Start background work AFTER app is created ----
    with app.app_context():
        start_background_tasks()
        schedule_daily_reddit_article(app)

    # IMPORTANT: return the Flask app
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
