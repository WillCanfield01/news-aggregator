# patchpal/app.py
import os
from pathlib import Path
from flask import Flask, request, render_template
from jinja2 import ChoiceLoader, FileSystemLoader
from slack_bolt import App as BoltApp
from slack_bolt.adapter.flask import SlackRequestHandler
from apscheduler.schedulers.background import BackgroundScheduler
from slack_sdk import WebClient

from .billing import billing_bp
from .storage import Base, engine
from .commands import register_commands
from .scheduler import run_once

# ---- Paths ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # repo root (has 'app/' and 'patchpal/')
PP_TEMPLATES = HERE / "templates"        # patchpal/templates
MAIN_TEMPLATES = ROOT / "app" / "templates"  # app/templates

# ---- DB init -------------------------------------------------------------
Base.metadata.create_all(engine)

# ---- Flask app -----------------------------------------------------------
flask_app = Flask(__name__, template_folder=str(PP_TEMPLATES))
# Let Jinja search both locations (PatchPal first, then main site)
flask_app.jinja_loader = ChoiceLoader([
    FileSystemLoader(str(PP_TEMPLATES)),
    FileSystemLoader(str(MAIN_TEMPLATES)),
])

flask_app.secret_key = os.getenv("FLASK_SECRET", "dev")

# Stripe / billing routes
flask_app.register_blueprint(billing_bp, url_prefix="/billing")

# ---- Slack Bolt ----------------------------------------------------------
signing_secret = os.getenv("SLACK_SIGNING_SECRET")
bot_token = os.getenv("SLACK_BOT_TOKEN")
bolt = BoltApp(signing_secret=signing_secret, token=bot_token)
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
client = WebClient(token=bot_token)
scheduler.add_job(lambda: run_once(client), "interval", minutes=1,
                  id="pp_tick", replace_existing=True, coalesce=True)
scheduler.start()

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
