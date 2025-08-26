# patchpal/app.py
import os
import logging
from pathlib import Path
from flask import Flask, request, render_template
from jinja2 import ChoiceLoader, FileSystemLoader
from apscheduler.schedulers.background import BackgroundScheduler
from slack_bolt import App as BoltApp
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk import WebClient

# ---------- paths / templates ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PP_TEMPLATES = HERE / "templates"
MAIN_TEMPLATES = ROOT / "app" / "templates"

# ---------- flask app (create FIRST) ----------
flask_app = Flask(__name__, template_folder=str(PP_TEMPLATES))
flask_app.secret_key = os.getenv("FLASK_SECRET", "dev")
flask_app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(PP_TEMPLATES)),
    FileSystemLoader(str(MAIN_TEMPLATES)),
])

# ---------- database (storage) ----------
# Uses your existing SQLAlchemy engine/Base from storage.py
from .storage import Base, engine  # noqa: E402
Base.metadata.create_all(engine)

# ---------- optional: billing ----------
try:
    from .billing import billing_bp  # noqa: E402
    flask_app.register_blueprint(billing_bp, url_prefix="/billing")
except Exception as e:
    logging.warning("Billing blueprint not loaded: %s", e)

# ---------- Slack OAuth blueprint ----------
# We try app_oauth first; fall back to oauth.py if that’s what you named it.
slack_oauth_bp = None
try:
    from .app_oauth import bp as slack_oauth_bp  # noqa: E402
except Exception:
    try:
        from .oauth import bp as slack_oauth_bp  # noqa: E402
    except Exception as e:
        logging.warning("Slack OAuth blueprint not loaded: %s", e)

if slack_oauth_bp:
    flask_app.register_blueprint(slack_oauth_bp)

# ---------- Slack Bolt (events & commands) ----------
signing_secret = os.getenv("SLACK_SIGNING_SECRET")
default_bot_token = os.getenv("SLACK_BOT_TOKEN")  # used for single-tenant/dev

# For public distribution you should resolve tokens per-workspace (see iter_bot_tokens below).
bolt = BoltApp(signing_secret=signing_secret, token=default_bot_token)

from .commands import register_commands  # noqa: E402
register_commands(bolt)

handler = SlackRequestHandler(bolt)

@flask_app.post("/slack/events")
def slack_events():
    # Slack sends events here; Bolt verifies the signature automatically
    return handler.handle(request)

# ---------- basic routes ----------
@flask_app.get("/legal")
def legal():
    return render_template("legal.html")

@flask_app.get("/healthz")
def health():
    return {"ok": True}

# ---------- per-workspace tokens ----------
def iter_bot_tokens():
    """
    Yield (team_id, bot_token) for all installations.
    Falls back to SLACK_BOT_TOKEN for single-tenant/dev if models aren’t present.
    """
    try:
        from .models import Installation, db  # expects a table with team_id, bot_token
        rows = db.session.query(Installation).all()
        yielded = False
        for row in rows:
            if getattr(row, "bot_token", None):
                yielded = True
                yield row.team_id, row.bot_token
        if not yielded and default_bot_token:
            # no installs yet, but we can still run with a single token in dev
            yield "default", default_bot_token
    except Exception as e:
        logging.warning("iter_bot_tokens(): falling back to env token (%s)", e)
        if default_bot_token:
            yield "default", default_bot_token

# ---------- scheduler ----------
from .scheduler import run_once  # noqa: E402

def tick():
    """
    Fan-out one run to each installed workspace.
    If your run_once(client) accepts a team_id, pass it along.
    """
    for team_id, tok in iter_bot_tokens():
        try:
            client = WebClient(token=tok)
            # If you later change run_once signature to run_once(client, team_id)
            # just add team_id=team_id here.
            run_once(client)
        except Exception as e:
            logging.exception("tick(): error running for team %s: %s", team_id, e)

# Start the scheduler only in the web dyno/process
if os.getenv("DISABLE_SCHEDULER", "0") not in ("1", "true", "yes"):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        tick,
        "interval",
        minutes=1,
        id="pp_tick",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()

# ---------- entrypoint ----------
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
