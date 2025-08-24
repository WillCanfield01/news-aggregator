# app/patchpal/commands.py
import re
from slack_bolt import App
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from .storage import SessionLocal, Workspace

HELP = (
    "*PatchPal commands:*\n"
    "• `/patchpal set-channel #channel`\n"
    "• `/patchpal set-time 09:00` (HH:MM 24h)\n"
    "• `/patchpal status`\n"
    "• `/patchpal post-now` (test a post immediately)\n"
)

# Strict HH:MM 24h (00–23 : 00–59)
TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")

# Slack formats a channel mention in slash-command text as `<#C12345678|channel-name>`
CHANNEL_MENTION_RE = re.compile(r"<#([A-Z0-9]+)\|[^>]+>", re.IGNORECASE)

def register_commands(app: App):
    @app.command("/patchpal")
    def patchpal_cmd(ack, body, respond, client, logger):
        # ACK immediately so Slack doesn't timeout (3s)
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
                    m = CHANNEL_MENTION_RE.search(text)
                    if not m:
                        respond(
                            "Please *mention* the channel so I get its ID, e.g.:\n"
                            "`/patchpal set-channel #security` (select the channel from the menu as you type)"
                        )
                        return
                    ws.post_channel = m.group(1)
                    db.commit()
                    respond(f"Got it. I’ll post in <#{ws.post_channel}> at {ws.post_time}.")
                    return

                if text.startswith("set-time"):
                    parts = text.split()
                    # allow `/patchpal set-time 9:00` or `/patchpal set-time 09:00`
                    hhmm = parts[-1] if len(parts) >= 2 else ""
                    if not TIME_RE.match(hhmm):
                        respond("Use HH:MM 24h, e.g., `09:00` or `9:00`.")
                        return
                    # zero-pad to HH:MM
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
                        respond("Set a channel first: `/patchpal set-channel #security`")
                        return
                    # lazy import to avoid circulars at module load
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

                # default help
                respond(HELP)

        except OperationalError as e:
            # transient DB/network issue — ask user to retry
            logger.warning(f"DB OperationalError: {e}")
            respond("Database connection hiccup — please run the command again in a few seconds.")
        except SQLAlchemyError as e:
            logger.exception("DB error")
            respond("Database error. Try again shortly; if it persists, ping support.")
        except Exception as e:
            logger.exception("Slash command failed")
            respond(f"Sorry, something went wrong: `{e.__class__.__name__}`. Try again or use `status`.")
