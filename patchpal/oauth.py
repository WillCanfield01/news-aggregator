# patchpal/oauth.py
import os, secrets, requests
from flask import Blueprint, redirect, request, session
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.web import WebClient

from .install_store import upsert_installation
from .storage import SessionLocal, Workspace
from .billing import ensure_trial

bp = Blueprint("slack_oauth", __name__)

APP_BASE_URL = (os.getenv("APP_BASE_URL") or "").rstrip("/")
if not APP_BASE_URL:
    raise RuntimeError("APP_BASE_URL is not set")

CLIENT_ID     = os.environ["SLACK_CLIENT_ID"]
CLIENT_SECRET = os.environ["SLACK_CLIENT_SECRET"]
REDIRECT_URI  = f"{APP_BASE_URL}/slack/oauth/callback"

# Add scopes you actually use
SCOPES = [
    "chat:write",
    "channels:read",
    "groups:read",
    "commands",
    "channels:join",     # for conversations.join
    "im:write",          # for conversations.open (DMs)
    "chat:write.public", # optional, post to public channels w/o joining
    "users:read",        # optional, nicer UX later
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
    # single-use state
    expected = session.pop("oauth_state", None)
    if request.args.get("state") != expected or not expected:
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

    team      = r.get("team") or {}
    team_id   = team.get("id")
    team_name = team.get("name", "")
    bot_token = r.get("access_token")     # xoxb-...
    bot_user  = r.get("bot_user_id")
    scopes    = r.get("scope") or ""
    installer = (r.get("authed_user") or {}).get("id")
    enterprise_id = (r.get("enterprise") or {}).get("id")

    # Persist installation
    upsert_installation(
        team_id=team_id,
        team_name=team_name,
        bot_token=bot_token,
        bot_user=bot_user,
        scopes=scopes,
        installed_by_user_id=installer,
        enterprise_id=enterprise_id,
    )

    # Ensure workspace row exists and start trial immediately
    try:
        with SessionLocal() as db:
            ws = db.query(Workspace).filter_by(team_id=team_id).first()
            if not ws:
                ws = Workspace(team_id=team_id, team_name=team_name, contact_user_id=installer)
                db.add(ws); db.commit()
                ensure_trial(ws, db)
            else:
                # keep contact up-to-date
                if installer and ws.contact_user_id != installer:
                    ws.contact_user_id = installer
                    db.commit()
    except Exception:
        # Non-fatal: installation is saved, just skip bootstrap if DB hiccups
        pass

    # Optional: DM the installer (requires im:write)
    try:
        if installer and bot_token:
            slack = WebClient(token=bot_token)
            im = slack.conversations_open(users=installer)
            slack.chat_postMessage(
                channel=im["channel"]["id"],
                text=f"✅ PatchPal installed in *{team_name}*.\n"
                     "Run `/patchpal set-channel here` in the target channel to start daily posts.\n"
                     "_For private channels: invite me with `/invite @PatchPal`._"
            )
    except Exception:
        pass

    return "App installed! You can close this window."
