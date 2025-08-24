# app/patchpal/commands.py
import re
from slack_bolt import App
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from .storage import SessionLocal, Workspace

HELP = (
    "*PatchPal commands:*\n"
    "• `/patchpal set-channel here`  (use the current channel)\n"
    "• `/patchpal set-channel #channel`  (or mention it; requires Slack setting or scopes)\n"
    "• `/patchpal set-time 09:00`  (HH:MM 24h)\n"
    "• `/patchpal status`\n"
    "• `/patchpal post-now`  (test a post immediately)\n"
)

TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")
MENTION_RE = re.compile(r"<#([A-Z0-9]+)\|[^>]+>", re.IGNORECASE)     # <#C123|name>
RAW_ID_RE  = re.compile(r"\b[CG][A-Z0-9]{8,}\b")                    # C123…, G123…
HASH_NAME_RE = re.compile(r"#([a-z0-9._-]+)", re.I)                 # #channel-name

def _resolve_channel_id(text: str, body: dict, client, logger):
    """Return a channel ID or None. Supports: here, <#C|…>, raw C/G ID, #name (best-effort)."""
    t = (text or "").strip()

    # 1) 'here' → use the channel where the command was invoked
    if t.lower().endswith("here"):
        return body.get("channel_id")

    # 2) Encoded mention <#C123|name>
    m = MENTION_RE.search(t)
    if m:
        return m.group(1)

    # 3) Raw channel ID in text
    m = RAW_ID_RE.search(t)
    if m:
        return m.group(0)

    # 4) Fallback: #channel-name → try to resolve via conversations.list (needs channels:read,groups:read)
    m = HASH_NAME_RE.search(t)
    if not m:
        return None
    name = m.group(1).lower()
    try:
        cursor = None
        while True:
            resp = client.conversations_list(
                types="public_channel,private_channel", limit=1000, cursor=cursor
            )
            for ch in resp.get("channels", []):
                nm = (ch.get("name_normalized") or ch.get("name") or "").lower()
                if nm == name:
                    return ch.get("id")
            cursor = resp.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                break
    except Exception as e:
        logger.info(f"conversations.list unavailable or missing scopes: {e}")
        # user can either enable “Escape channels…” on the Slash Command,
        # or use `/patchpal set-channel here`
        return None

def register_commands(app: App):
    @app.command("/patchpal")
    def patchpal_cmd(ack, body, respond, client, logger):
        # ACK immediately (3s SLA)
        ack()

        team_id = body.get("team_id")
        text = (body.get("text") or "").strip()

        try:
            with SessionLocal() as db:
                ws = db.query(Workspace).filter_by(team_id=team_id).first()
                if not ws:
                    ws = Workspace(team_id=team_id)
                    db.add(ws)
                    db.commit()

                if text.startswith("set-channel"):
                    cid = _resolve_channel_id(text, body, client, logger)
                    if not cid:
                        respond(
                            "Please *mention* the channel (so Slack sends `<#C…|name>`) "
                            "or just run `*/patchpal set-channel here*` in the target channel.\n"
                            "_Tip: In your Slack app’s Slash Command settings, enable "
                            "“Escape channels, users, and links sent to your app.”_"
                        )
                        return
                    ws.post_channel = cid
                    db.commit()
                    respond(f"Got it. I’ll post in <#{ws.post_channel}> at {ws.post_time}.")
                    return

                if text.startswith("set-time"):
                    parts = text.split()
                    hhmm = parts[-1] if len(parts) >= 2 else ""
                    if not TIME_RE.match(hhmm):
                        respond("Use HH:MM 24h, e.g., `09:00` or `9:00`.")
                        return
                    h, m = hhmm.split(":")
                    ws.post_time = f"{int(h):02d}:{int(m):02d}"
                    db.commit()
                    respond(f"Time set to {ws.post_time}.")
                    return

                if text.startswith("status"):
                    ch = f"<#{ws.post_channel}>" if ws.post_channel else "_not set_"
                    respond(f"Channel: {ch}  |  Time: {ws.post_time}  |  TZ: {ws.tz}")
                    return

                if text.startswith("post-now"):
                    if not ws.post_channel:
                        respond("Set a channel first: `*/patchpal set-channel here*` (run it in the target channel).")
                        return
                    from .selector import top3_today, as_slack_blocks
                    items = top3_today()
                    if not items:
                        respond("No items found right now.")
                        return
                    client.chat_postMessage(
                        channel=ws.post_channel,
                        text="Today’s Top 3 Patches / CVEs",
                        blocks=as_slack_blocks(items),
                    )
                    respond(f"Posted to <#{ws.post_channel}>.")
                    return

                respond(HELP)

        except OperationalError as e:
            logger.warning(f"DB OperationalError: {e}")
            respond("Database connection hiccup — please run the command again in a few seconds.")
        except SQLAlchemyError as e:
            logger.exception("DB error")
            respond("Database error. Try again shortly; if it persists, ping support.")
        except Exception as e:
            logger.exception("Slash command failed")
            respond(f"Sorry, something went wrong: `{e.__class__.__name__}`. Try again or use `status`.")
