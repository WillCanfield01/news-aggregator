# app/patchpal/commands.py
import re
from slack_bolt import App
from .storage import SessionLocal, Workspace

HELP = (
    "*PatchPal commands:*\n"
    "• `/patchpal set-channel #channel`\n"
    "• `/patchpal set-time 09:00` (HH:MM 24h)\n"
    "• `/patchpal status`\n"
)

def register_commands(app: App):
    @app.command("/patchpal")
    def patchpal_cmd(ack, body, respond, client, logger):
        # ACK immediately so Slack doesn't timeout (3s SLA)
        ack()

        try:
            team_id = body.get("team_id")
            text = (body.get("text") or "").strip()

            db = SessionLocal()
            ws = db.query(Workspace).filter_by(team_id=team_id).first()
            if not ws:
                # Keep it simple—no team.info call (avoids team:read scope)
                ws = Workspace(team_id=team_id, team_name=None)
                db.add(ws)
                db.commit()

            if text.startswith("set-channel"):
                m = re.search(r"<#(\w+)\|", text)
                if not m:
                    respond("Please mention a channel like `#security`.")
                else:
                    ws.post_channel = m.group(1)
                    db.commit()
                    respond(f"Got it. I’ll post in <#{ws.post_channel}> at {ws.post_time}.")
            elif text.startswith("set-time"):
                m = re.search(r"(\d{2}):(\d{2})", text)
                if not m:
                    respond("Use HH:MM 24h, e.g., `09:00`.")
                else:
                    ws.post_time = f"{m.group(1)}:{m.group(2)}"
                    db.commit()
                    respond(f"Time set to {ws.post_time}.")
            elif text.startswith("status"):
                respond(f"Channel: <#{ws.post_channel}>  |  Time: {ws.post_time}  |  TZ: {ws.tz}")
            else:
                respond(HELP)
        except Exception as e:
            logger.exception("Slash command failed")
            respond(f"Sorry, something went wrong: `{e.__class__.__name__}`. Try again or `status`.")
        finally:
            try:
                db.close()
            except Exception:
                pass