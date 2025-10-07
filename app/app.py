# app/app.py
import os, re
from datetime import datetime
from flask import Flask, render_template, Response, url_for, current_app, jsonify, redirect
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
from app.roulette import models as _roulette_models 
from app.extensions import db, login_manager
# escape feature
from app.escape.core import schedule_daily_generation
from app.escape import create_escape_bp
from app.scripts.generate_timeline_round import ensure_today_round

def schedule_daily_reddit_article(app):
    def scheduled_job():
        with app.app_context():
            from app.reddit_articles import generate_article_for_today
            generate_article_for_today()
            print("✅ Daily Reddit article generated at", datetime.now())

    scheduler = BackgroundScheduler(timezone=pytz.timezone("America/Denver"))
    scheduler.add_job(scheduled_job, "cron", hour=17, minute=0)
    scheduler.start()

def schedule_daily_timeline(app):
    def job():
        with app.app_context():
            ensure_today_round()
            print("✅ Timeline Roulette generated")
    scheduler = BackgroundScheduler(timezone=pytz.timezone("America/Denver"))
    scheduler.add_job(job, "cron", hour=0, minute=5)  # adjust to your TZ
    scheduler.start()

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ---- Configs ----
    db_url = os.environ.get("DATABASE_URL", "sqlite:///local.db")

    # Normalize to psycopg v3 driver
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+psycopg://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
    app.config["PATCHPAL_LEGAL_URL"] = os.getenv("PATCHPAL_LEGAL_URL", "https://your-patchpal-host/legal")
    app.secret_key = os.environ.get("SECRET_KEY", "super-secret-dev-key")
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

    # ---- Bind extensions FIRST ----
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "aggregator.login"

    # ---- Import models so SQLAlchemy knows them, then (optionally) create tables ----
    from app.escape import models as _escape_models  # noqa: F401
    from app.models import CommunityArticle          # site models
    with app.app_context():
        db.create_all()

    # ---- Blueprints ----
    from app.aggregator import aggregator_bp, start_background_tasks
    from app.reddit_articles import bp as reddit_bp
    from app.roulette import roulette_bp
    app.register_blueprint(aggregator_bp)
    app.register_blueprint(reddit_bp)
    app.register_blueprint(create_escape_bp(), url_prefix="/escape")
    app.register_blueprint(roulette_bp)    

    # Optional: /escape → /escape/today convenience
    @app.route("/escape")
    def escape_root_redirect():
        return redirect("/escape/today", code=302)

    @app.route("/_roulette_ping")
    def _roulette_ping():
        return "roulette blueprint loaded"

    # ---- Routes ----
    @app.route("/")
    def landing():
        latest = CommunityArticle.query.order_by(CommunityArticle.date.desc()).first()
        return render_template("index.html", year=datetime.now().year, latest=latest)

    @app.route("/sitemap.xml")
    def sitemap():
        from flask import make_response
        try:
            articles = (
                CommunityArticle.query
                .order_by(CommunityArticle.date.desc(), CommunityArticle.id.desc())
                .all()
            )
            latest_date = (articles[0].date if articles and articles[0].date else None)

            urls = []
            urls.append(f"""<url>
<loc>{url_for('landing', _external=True, _scheme='https')}</loc>
{f"<lastmod>{latest_date.strftime('%Y-%m-%d')}</lastmod>" if latest_date else ""}
<changefreq>daily</changefreq><priority>1.0</priority>
</url>""")

            urls.append(f"""<url>
<loc>{url_for('all_articles.published_articles', _external=True, _scheme='https')}</loc>
{f"<lastmod>{latest_date.strftime('%Y-%m-%d')}</lastmod>" if latest_date else ""}
<changefreq>daily</changefreq><priority>0.9</priority>
</url>""")

            for a in articles:
                loc = url_for("all_articles.read_article", filename=a.filename, _external=True, _scheme='https')
                lastmod = a.date.strftime("%Y-%m-%d") if a.date else ""
                urls.append(f"""<url>
<loc>{loc}</loc>
{f"<lastmod>{lastmod}</lastmod>" if lastmod else ""}
<changefreq>weekly</changefreq><priority>0.8</priority>
</url>""")

            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{''.join(urls)}
</urlset>"""

            resp = make_response(xml, 200)
            resp.headers["Content-Type"] = "application/xml; charset=utf-8"
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp
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
        schedule_daily_generation(app)
        schedule_daily_timeline(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
