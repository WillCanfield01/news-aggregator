# patchpal/app.py
import os
from pathlib import Path

from flask import Flask, request, render_template, jsonify
from jinja2 import ChoiceLoader, FileSystemLoader
from apscheduler.schedulers.background import BackgroundScheduler
from slack_bolt import App as BoltApp
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.authorization import AuthorizeResult
from slack_sdk import WebClient

from .billing import billing_bp
from .commands import register_commands
from .oauth import bp as slack_oauth_bp  # /slack/install + /slack/oauth/callback
from .storage import Base, engine, SessionLocal, Workspace, delete_recent_for_team
from .install_store import get_bot_token, migrate_file_store_if_present, delete_install

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
flask_app.register_blueprint(slack_oauth_bp)

# ---- Slack Bolt (multi-workspace) ----------------------------------------
signing_secret = os.getenv("SLACK_SIGNING_SECRET")

def authorize(enterprise_id, team_id, user_id=None, client=None, logger=None, **kwargs):
    # Optional single-team dev fallback
    token = get_bot_token(team_id) or os.getenv("SLACK_BOT_TOKEN")
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

# ---- Admin: Delete workspace + data --------------------------------------
ADMIN_DELETE_TOKEN = os.getenv("ADMIN_DELETE_TOKEN", "")

@flask_app.post("/admin/delete_workspace")
def delete_workspace():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not ADMIN_DELETE_TOKEN or token != ADMIN_DELETE_TOKEN:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    team_id = (request.json or {}).get("team_id") or request.args.get("team_id")
    if not team_id:
        return jsonify({"ok": False, "error": "missing team_id"}), 400

    with SessionLocal() as db:
        ws = db.query(Workspace).filter_by(team_id=team_id).first()
        if ws:
            db.delete(ws)
            db.commit()

    delete_install(team_id)
    delete_recent_for_team(team_id)

    return jsonify({"ok": True})

# ---- Scheduler -----------------------------------------------------------
scheduler = BackgroundScheduler()

PRIMARY_TEAM = os.getenv("SLACK_PRIMARY_TEAM")  # optional single-team tick token
_sched_token = get_bot_token(PRIMARY_TEAM) or os.getenv("SLACK_BOT_TOKEN")
_client = WebClient(token=_sched_token) if _sched_token else None

def _tick():
    """Lazy-import the scheduler each tick to avoid import-time cycles."""
    if _client is None:
        # No token configured for the background poster; skip.
        return
    try:
        from .scheduler import run_once as _run_once
        _run_once(_client)
    except Exception as e:
        print(f"[app] scheduler tick failed: {e}")

# Allow disabling the scheduler on certain dynos/containers
if os.getenv("ENABLE_SCHEDULER", "1").lower() in ("1", "true", "yes"):
    scheduler.add_job(_tick, "interval", minutes=1,
                      id="pp_tick", replace_existing=True, coalesce=True)
    scheduler.start()

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
