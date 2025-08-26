# patchpal/oauth.py
import os, secrets, requests
from flask import Blueprint, redirect, request, session
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.web import WebClient

bp = Blueprint("slack_oauth", __name__)

CLIENT_ID     = os.environ["SLACK_CLIENT_ID"]
CLIENT_SECRET = os.environ["SLACK_CLIENT_SECRET"]
REDIRECT_URI  = os.getenv("APP_BASE_URL").rstrip("/") + "/slack/oauth/callback"

SCOPES = [
    "chat:write",      # post messages
    "channels:read",   # optional: browse public channels for a picker
    "groups:read",     # optional: browse private channels user has access to
    # add "commands" later if you ship a slash command
]

authz = AuthorizeUrlGenerator(
    client_id=CLIENT_ID, scopes=SCOPES, redirect_uri=REDIRECT_URI
)

@bp.get("/slack/install")
def slack_install():
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    return redirect(authz.generate(state=state))

@bp.get("/slack/oauth/callback")
def slack_oauth_callback():
    if request.args.get("state") != session.get("oauth_state"):
        return "Invalid state", 400

    code = request.args.get("code")
    r = requests.post(
        "https://slack.com/api/oauth.v2.access",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=10,
    ).json()
    if not r.get("ok"):
        return f"OAuth error: {r}", 400

    team_id   = r["team"]["id"]
    team_name = r["team"]["name"]
    bot_token = r["access_token"]     # xoxb-...
    bot_user  = r["bot_user_id"]
    authed_user = r.get("authed_user", {}).get("id")

    # <<< save install >>>
    save_installation(team_id, team_name, bot_token, bot_user)

    # Nice touch: DM the installer (falls back if we can’t)
    try:
        if authed_user:
            WebClient(token=bot_token).chat_postMessage(
                channel=authed_user,
                text=f"✅ PatchPal installed in *{team_name}*. Invite me to a channel and I’ll start posting."
            )
    except Exception:
        pass

    return "App installed! You can close this window."

# --- storage hook (swap to your DB) ---
def save_installation(team_id, team_name, bot_token, bot_user):
    """
    Replace this with your DB logic. It must upsert by team_id.
    """
    # Example using a simple JSON file for first run; replace with SQLAlchemy.
    import json, pathlib
    p = pathlib.Path("/mnt/data/installations.json")
    store = {}
    if p.exists():
        store = json.loads(p.read_text())
    store[team_id] = {
        "team_name": team_name,
        "bot_token": bot_token,
        "bot_user": bot_user,
    }
    p.write_text(json.dumps(store))
