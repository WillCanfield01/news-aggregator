import os
from flask import Flask, request
from slack_bolt import App as BoltApp
from slack_bolt.adapter.flask import SlackRequestHandler
from apscheduler.schedulers.background import BackgroundScheduler
from slack_sdk import WebClient

from storage import Base, engine
from commands import register_commands
from scheduler import run_once

# DB init
Base.metadata.create_all(engine)

# Slack Bolt (single-workspace for MVP)
signing_secret = os.getenv("SLACK_SIGNING_SECRET")
bot_token = os.getenv("SLACK_BOT_TOKEN")
bolt = BoltApp(signing_secret=signing_secret, token=bot_token)
register_commands(bolt)
handler = SlackRequestHandler(bolt)

# Flask
flask_app = Flask(__name__)
flask_app.secret_key = os.getenv("FLASK_SECRET", "dev")

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.get("/healthz")
def health():
    return {"ok": True}

# Scheduler: tick every minute
scheduler = BackgroundScheduler()
client = WebClient(token=bot_token)
scheduler.add_job(lambda: run_once(client), "interval", minutes=1, id="pp_tick")
scheduler.start()

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
