# patchpal/scheduler.py
from datetime import datetime
import pytz
from slack_sdk import WebClient

from .storage import SessionLocal, Workspace, PostLog
from .selector import topN_today, render_item_text
from .billing import (
    is_active,
    ensure_trial,
    dm_trial_or_checkout,
    dm_trial_ending_soon,   # ← pre-expiry reminder
)

DEF_TZ = "America/Boise"


def should_post_now(tz_str: str | None, hhmm: str | None) -> bool:
    """Return True exactly at HH:MM (workspace tz), Mon–Fri."""
    try:
        tz = pytz.timezone(tz_str or DEF_TZ)
    except Exception:
        tz = pytz.timezone(DEF_TZ)

    now = datetime.now(tz)
    try:
        h, m = map(int, (hhmm or "09:00").split(":"))
    except Exception:
        h, m = 9, 0

    return now.weekday() < 5 and now.hour == h and now.minute == m


def run_once(client: WebClient) -> None:
    """One tick of the scheduler. Safe to call every minute."""
    today = datetime.utcnow().strftime("%Y-%m-%d")

    with SessionLocal() as db:
        teams = db.query(Workspace).all()

        for ws in teams:
            # Ensure every workspace has a trial start date
            ensure_trial(ws, db)

            # 🔔 Pre-expiry reminder (3 days before trial end). Non-blocking.
            try:
                dm_trial_ending_soon(client, ws)
            except Exception:
                # Never let reminder failures break posting
                pass

            # Only proceed at the configured time (Mon–Fri) and if a channel is set
            if not (ws.post_channel and should_post_now(ws.tz, ws.post_time)):
                continue

            # One post per day per workspace
            if db.query(PostLog).filter_by(team_id=ws.team_id, post_date=today).first():
                continue

            # 🔒 Billing gate: pause & DM upgrade link if inactive
            if not is_active(ws=ws):
                dm_trial_or_checkout(client, ws)
                continue

            # Build today's items
            items = topN_today(5, ws=ws)  # always 5
            if not items:
                continue

            try:
                # Header message
                hdr = client.chat_postMessage(
                    channel=ws.post_channel,
                    text="Today’s Top 5 Patches / CVEs",
                    blocks=[{
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Today’s Top 5 Patches / CVEs", "emoji": True},
                    }],
                )
                parent_ts = hdr["ts"]

                # Items in thread (keeps long text from being truncated)
                tone = (ws.tone or "simple")
                for i, it in enumerate(items, 1):
                    txt = render_item_text(it, i, tone)
                    client.chat_postMessage(
                        channel=ws.post_channel,
                        thread_ts=parent_ts,
                        text=f"{i})",  # fallback
                        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": txt}}],
                    )

                # Footer context
                client.chat_postMessage(
                    channel=ws.post_channel,
                    thread_ts=parent_ts,
                    text="",
                    blocks=[{
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal · by The Real Roundup"}],
                    }],
                )

                # Mark success
                db.add(PostLog(team_id=ws.team_id, post_date=today))
                db.commit()

            except Exception as e:
                print(f"[scheduler] post failed for team {ws.team_id}: {e}")
