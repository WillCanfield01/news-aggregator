# patchpal/app.py
import os
from pathlib import Path
from flask import Flask, request, render_template
from jinja2 import ChoiceLoader, FileSystemLoader
from slack_bolt import App as BoltApp
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.authorization import AuthorizeResult
from apscheduler.schedulers.background import BackgroundScheduler
from slack_sdk import WebClient
from .install_store import get_bot_token, migrate_file_store_if_present
from .billing import billing_bp
from .storage import Base, engine
from .commands import register_commands
from .scheduler import run_once  # uses selector, no circular import
from .oauth import bp as slack_oauth_bp          # your OAuth blueprint
from .install_store import get_bot_token, migrate_file_store_if_present

# ---- Paths ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PP_TEMPLATES = HERE / "templates"
MAIN_TEMPLATES = ROOT / "app" / "templates"

# ---- DB init -------------------------------------------------------------
Base.metadata.create_all(engine)
migrate_file_store_if_present()
# ---- Flask app -----------------------------------------------------------
flask_app = Flask(__name__, template_folder=str(PP_TEMPLATES))
flask_app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(PP_TEMPLATES)),
    FileSystemLoader(str(MAIN_TEMPLATES)),
])
flask_app.secret_key = os.getenv("FLASK_SECRET", "dev")

# Blueprints
flask_app.register_blueprint(billing_bp, url_prefix="/billing")
flask_app.register_blueprint(slack_oauth_bp)  # /slack/install + /slack/oauth/callback

# ---- Slack Bolt (multi-workspace) ----------------------------------------
signing_secret = os.getenv("SLACK_SIGNING_SECRET")

def authorize(enterprise_id, team_id, user_id, client, logger):
    token = get_bot_token(team_id)
    # (Optional dev fallback: only if you truly want single-team dev)
    if not token:
        token = os.getenv("SLACK_BOT_TOKEN")  # remove in production
    return AuthorizeResult(
        enterprise_id=enterprise_id,
        team_id=team_id,
        user_id=user_id,
        bot_token=token,
    )

bolt = BoltApp(signing_secret=signing_secret, authorize=authorize)
register_commands(bolt)
handler = SlackRequestHandler(bolt)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.get("/legal")
def legal():
    return render_template("legal.html")

@flask_app.get("/healthz")
def health():
    return {"ok": True}

# ---- Scheduler -----------------------------------------------------------
scheduler = BackgroundScheduler()

# choose a token for scheduled posts:
PRIMARY_TEAM = os.getenv("SLACK_PRIMARY_TEAM")  # optional
_sched_token = get_bot_token(PRIMARY_TEAM) or os.getenv("SLACK_BOT_TOKEN")
client = WebClient(token=_sched_token)

scheduler.add_job(lambda: run_once(client), "interval", minutes=1,
                  id="pp_tick", replace_existing=True, coalesce=True)
scheduler.start()

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
