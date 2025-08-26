# patchpal/app.py
import os
from pathlib import Path
from flask import Flask, request, render_template
from jinja2 import ChoiceLoader, FileSystemLoader

# ---- Paths ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PP_TEMPLATES = HERE / "templates"
MAIN_TEMPLATES = ROOT / "app" / "templates"

# ---- Flask app (create first!) -------------------------------------------
flask_app = Flask(__name__, template_folder=str(PP_TEMPLATES))
flask_app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(PP_TEMPLATES)),
    FileSystemLoader(str(MAIN_TEMPLATES)),
])
flask_app.secret_key = os.getenv("FLASK_SECRET", "dev")

# ---- DB / models ----------------------------------------------------------
from .storage import Base, engine  # uses your existing engine
Base.metadata.create_all(engine)   # ensures tables (including installations) exist

# ---- Blueprints -----------------------------------------------------------
# Your OAuth blueprint should define: bp = Blueprint("slack_oauth", __name__)
# and routes for /slack/install and /slack/oauth/callback
from .oauth import bp as slack_oauth_bp
flask_app.register_blueprint(slack_oauth_bp, url_prefix="/slack")

# ---- Slack (Bolt) ---------------------------------------------------------
# We support per-workspace tokens via authorize(), so no global SLACK_BOT_TOKEN is required.
from slack_bolt import App as BoltApp
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.authorization import AuthorizeResult
from .models import get_bot_token  # looks up tokens saved by OAuth

signing_secret = os.getenv("SLACK_SIGNING_SECRET")

def authorize(enterprise_id, team_id, user_id, client, logger):
    """
    Provide a token dynamically per workspace (team_id).
    Bolt will call this for each request that needs a token.
    """
    token = get_bot_token(team_id) if team_id else None
    if token:
        return AuthorizeResult(
            enterprise_id=enterprise_id,
            team_id=team_id,
            bot_token=token,
        )
    # No token yet (e.g., app not installed) — allow handlers to ack but not post
    return AuthorizeResult(enterprise_id=enterprise_id, team_id=team_id)

# If the signing secret is missing, disable the events route but keep the app booting.
bolt = None
handler = None
if signing_secret:
    bolt = BoltApp(signing_secret=signing_secret, authorize=authorize)
    # Register your commands/listeners *after* Bolt is created
    from .commands import register_commands
    register_commands(bolt)
    handler = SlackRequestHandler(bolt)
else:
    print("⚠️  SLACK_SIGNING_SECRET not set — /slack/events disabled")

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    if handler is None:
        return ("Slack events not configured", 503)
    return handler.handle(request)

# ---- Simple site routes ----------------------------------------------------
@flask_app.get("/legal")
def legal():
    return render_template("legal.html")

@flask_app.get("/healthz")
def health():
    return {"ok": True}

# ---- Scheduler (optional) --------------------------------------------------
# Running schedulers inside web dynos can cause dupes if >1 worker.
# Only enable when RUN_SCHEDULER=1, and keep WEB_CONCURRENCY=1 on Render.
if os.getenv("RUN_SCHEDULER", "0").lower() in ("1", "true", "yes"):
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from slack_sdk import WebClient
        from .scheduler import run_once

        def _get_client_for_default_team():
            """
            Optional: pick a default team for scheduled posts if your scheduler
            isn’t tied to a specific team. Otherwise, refactor run_once to loop over
            all installations and post with each token.
            """
            default_team = os.getenv("DEFAULT_TEAM_ID")
            token = get_bot_token(default_team) if default_team else None
            return WebClient(token=token) if token else None

        scheduler = BackgroundScheduler()
        def _tick():
            client = _get_client_for_default_team()
            if client:
                run_once(client)
            else:
                # No token available yet — skip silently
                pass

        scheduler.add_job(
            _tick, "interval", minutes=int(os.getenv("SCHED_INTERVAL_MIN", "5")),
            id="pp_tick", replace_existing=True, coalesce=True
        )
        scheduler.start()
        print("⏰ Scheduler started")
    except Exception as e:
        # Never crash the web process because the scheduler failed to init
        print(f"⚠️  Scheduler did not start: {e}")

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
