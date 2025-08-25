# patchpal/commands.py
import re
from slack_bolt import App
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from .storage import SessionLocal, Workspace
from .billing import ensure_trial  # start a 14-day trial on first touch

HELP = (
    "*PatchPal commands:*\n"
    "• `/patchpal set-channel here`  (or mention a channel)\n"
    "• `/patchpal set-time 09:00`  (HH:MM 24h)\n"
    "• `/patchpal set-tone simple|detailed`\n"
    "• `/patchpal status`\n"
    "• `/patchpal post-now`  (test immediately)\n"
)

# Regex helpers
TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")
MENTION_RE = re.compile(r"<#([A-Z0-9]+)\|[^>]+>", re.I)   # <#C123|name>
RAW_ID_RE  = re.compile(r"\b[CG][A-Z0-9]{8,}\b")         # C123... / G123...
HASH_NAME_RE = re.compile(r"#([a-z0-9._-]+)", re.I)      # #channel-name

def _resolve_channel_id(text: str, body: dict, client, logger):
    """Return a channel ID from 'here', <#C...|...>, raw ID, or #name (best-effort)."""
    t = (text or "").strip()

    # 1) 'here' → channel the command was invoked in
    if t.lower().endswith("here"):
        return body.get("channel_id")

    # 2) Encoded mention <#C123|name>
    m = MENTION_RE.search(t)
    if m:
        return m.group(1)

    # 3) Raw channel ID
    m = RAW_ID_RE.search(t)
    if m:
        return m.group(0)

    # 4) #channel-name → try conversations.list (needs channels:read,groups:read if private)
    m = HASH_NAME_RE.search(t)
    if not m:
        return None
    name = m.group(1).lower()
    try:
        cursor = None
        while True:
            resp = client.conversations_list(types="public_channel,private_channel", limit=1000, cursor=cursor)
            for ch in resp.get("channels", []):
                nm = (ch.get("name_normalized") or ch.get("name") or "").lower()
                if nm == name:
                    return ch.get("id")
            cursor = resp.get("response_metadata", {}).get("next_cursor") or None
            if not cursor:
                break
    except Exception as e:
        logger.info(f"conversations.list missing scopes or failed: {e}")
        return None

def register_commands(app: App):
    @app.command("/patchpal")
    def patchpal_cmd(ack, body, respond, client, logger):
        # ACK fast (3s window)
        ack()
        team_id = body.get("team_id")
        user_id = body.get("user_id")
        text = (body.get("text") or "").strip()

        try:
            with SessionLocal() as db:
                ws = db.query(Workspace).filter_by(team_id=team_id).first()
                if not ws:
                    ws = Workspace(team_id=team_id)
                    db.add(ws); db.commit()
                    ensure_trial(ws, db)  # start 14-day trial on first touch

                # always remember a contact user to DM for billing notices
                if user_id and ws.contact_user_id != user_id:
                    ws.contact_user_id = user_id
                    db.commit()

                # --- set-channel ---
                if text.startswith("set-channel"):
                    cid = _resolve_channel_id(text, body, client, logger)
                    if not cid:
                        respond(
                            "Please *mention* the channel (so Slack sends `<#C…|name>`) or run "
                            "`/patchpal set-channel here` in the target channel.\n"
                            "_Tip: In your Slack app, enable “Escape channels, users, and links sent to your app.”_"
                        )
                        return
                    ws.post_channel = cid
                    db.commit()
                    respond(f"Got it. I’ll post in <#{ws.post_channel}> at {ws.post_time}.")
                    return

                # --- set-time ---
                if text.startswith("set-time"):
                    parts = text.split()
                    hhmm = parts[-1] if len(parts) >= 2 else ""
                    if not TIME_RE.match(hhmm):
                        respond("Use HH:MM 24h, e.g., `09:00`.")
                        return
                    h, m = hhmm.split(":")
                    ws.post_time = f"{int(h):02d}:{int(m):02d}"
                    db.commit()
                    respond(f"Time set to {ws.post_time}.")
                    return

                # --- set-tone ---
                if text.startswith("set-tone"):
                    parts = text.split()
                    tone = (parts[-1] if len(parts) >= 2 else "").lower()
                    if tone not in ("simple", "detailed"):
                        respond("Use `simple` or `detailed`, e.g., `/patchpal set-tone simple`.")
                        return
                    ws.tone = tone
                    db.commit()
                    respond(f"Tone set to *{ws.tone}*.")
                    return

                # --- status ---
                if text.startswith("status"):
                    ch = f"<#{ws.post_channel}>" if ws.post_channel else "_not set_"
                    trial = f"{(ws.trial_ends_at.isoformat()[:10] if ws.trial_ends_at else 'n/a')}"
                    plan = ws.plan or "trial"
                    respond(
                        f"Channel: {ch}  |  Time: {ws.post_time}  |  Tone: *{ws.tone or 'simple'}*  "
                        f"|  Plan: *{plan}*  |  Trial ends: *{trial}*  |  TZ: {ws.tz}"
                    )
                    return

                # --- post-now (threaded) ---
                if text.startswith("post-now"):
                    if not ws.post_channel:
                        respond("Set a channel first: `*/patchpal set-channel here*` in the target channel.")
                        return

                    # Import here to avoid circulars and keep cold start fast
                    from .selector import topN_today, render_item_text

                    items = topN_today(5)  # always 5
                    if not items:
                        respond("No items found right now.")
                        return

                    # 1) Post a header in the channel
                    hdr = client.chat_postMessage(
                        channel=ws.post_channel,
                        text="Today’s Top 5 Patches / CVEs",
                        blocks=[{
                            "type": "header",
                            "text": {"type": "plain_text", "text": "Today’s Top 5 Patches / CVEs", "emoji": True}
                        }],
                    )
                    parent_ts = hdr["ts"]

                    # 2) Post each item as a thread reply (prevents Slack dropping long blocks)
                    for i, it in enumerate(items, 1):
                        txt = render_item_text(it, i, ws.tone or "simple")
                        client.chat_postMessage(
                            channel=ws.post_channel,
                            thread_ts=parent_ts,
                            text=f"{i})",  # fallback
                            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": txt}}],
                        )

                    # 3) Footer context
                    client.chat_postMessage(
                        channel=ws.post_channel,
                        thread_ts=parent_ts,
                        text="",
                        blocks=[{
                            "type": "context",
                            "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal · by The Real Roundup"}],
                        }],
                    )

                    respond(f"Posted 5 item(s) to <#{ws.post_channel}>.")
                    return

                # default help
                respond(HELP)

        except OperationalError as e:
            logger.warning(f"DB OperationalError: {e}")
            respond("Database connection hiccup — please try again in a few seconds.")
        except SQLAlchemyError as e:
            logger.exception("DB error")
            respond("Database error. Try again shortly.")
        except Exception as e:
            logger.exception("Slash command failed")
            respond(f"Sorry, something went wrong: `{e.__class__.__name__}`. Try again or use `status`.")
