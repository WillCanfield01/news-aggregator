# patchpal/oauth.py
import os, secrets, requests
from flask import Blueprint, redirect, request, session
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.web import WebClient
from .install_store import upsert_installation
from .selector import upsert_installation  # <- use the helper you added there

bp = Blueprint("slack_oauth", __name__)

APP_BASE_URL = (os.getenv("APP_BASE_URL") or "").rstrip("/")
if not APP_BASE_URL:
    raise RuntimeError("APP_BASE_URL is not set")

CLIENT_ID     = os.environ["SLACK_CLIENT_ID"]
CLIENT_SECRET = os.environ["SLACK_CLIENT_SECRET"]
REDIRECT_URI  = f"{APP_BASE_URL}/slack/oauth/callback"

SCOPES = [
    "chat:write",
    "channels:read",
    "groups:read",
    "commands",
]

authz = AuthorizeUrlGenerator(
    client_id=CLIENT_ID,
    scopes=SCOPES,
    redirect_uri=REDIRECT_URI,
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
    scopes    = r.get("scope") or ""
    installer = (r.get("authed_user") or {}).get("id")

    # persist installation
    enterprise_id = (r.get("enterprise") or {}).get("id")
    upsert_installation(
        team_id=team_id,
        team_name=team_name,
        bot_token=bot_token,
        bot_user=bot_user,
        scopes=scopes,
        installed_by_user_id=installer,
        enterprise_id=enterprise_id,
    )

    # optional: DM the installer
    try:
        if installer:
            WebClient(token=bot_token).chat_postMessage(
                channel=installer,
                text=f"✅ PatchPal installed in *{team_name}*. Invite me to a channel and I’ll start posting."
            )
    except Exception:
        pass

    return "App installed! You can close this window."
