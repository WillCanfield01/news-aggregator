# patchpal/scheduler.py
from __future__ import annotations

from datetime import datetime
import pytz
from slack_sdk import WebClient

from .install_store import get_bot_token
from .storage import SessionLocal, Workspace, PostLog
from .selector import topN_today, render_item_text

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

def _client_for(team_id: str) -> WebClient | None:
    tok = get_bot_token(team_id)
    return WebClient(token=tok) if tok else None

def run_once(_unused_client: WebClient | None = None) -> None:
    """One scheduler tick. Safe to run every minute; posts per-team with their own tokens."""
    # Lazy import to avoid any import-time circular refs
    from .billing import (
        is_active,
        ensure_trial,
        dm_trial_or_checkout,
        dm_trial_ending_soon,
    )

    today = datetime.utcnow().strftime("%Y-%m-%d")

    with SessionLocal() as db:
        teams = db.query(Workspace).all()

        for ws in teams:
            # Ensure every workspace has a trial start date (idempotent)
            try:
                ensure_trial(ws, db)
            except Exception:
                pass

            client = _client_for(ws.team_id)

            # Best-effort pre-expiry reminder (3 days prior)
            try:
                if client:
                    dm_trial_ending_soon(client, ws)
            except Exception:
                pass

            # Only at configured time on weekdays and if a channel is set
            if not (ws.post_channel and should_post_now(ws.tz, ws.post_time)):
                continue

            # One post per team per day
            if db.query(PostLog).filter_by(team_id=ws.team_id, post_date=today).first():
                continue

            # Billing gate (pause + DM if inactive)
            if not is_active(ws=ws):
                try:
                    if client:
                        dm_trial_or_checkout(client, ws)
                except Exception:
                    pass
                continue

            if not client:
                print(f"[scheduler] no token for team {ws.team_id}; skipping")
                continue

            # Build and post
            items = topN_today(5, ws=ws, team_id=ws.team_id)
            if not items:
                continue

            try:
                hdr = client.chat_postMessage(
                    channel=ws.post_channel,
                    text="Today’s Top 5 Patches / CVEs",
                    blocks=[{
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Today’s Top 5 Patches / CVEs", "emoji": True},
                    }],
                )
                parent_ts = hdr["ts"]

                tone = ws.tone or "simple"
                for i, it in enumerate(items, 1):
                    txt = render_item_text(it, i, tone)
                    client.chat_postMessage(
                        channel=ws.post_channel,
                        thread_ts=parent_ts,
                        text=f"{i})",  # fallback
                        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": txt}}],
                    )

                client.chat_postMessage(
                    channel=ws.post_channel,
                    thread_ts=parent_ts,
                    text="",
                    blocks=[{
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal · by The Real Roundup"}],
                    }],
                )

                db.add(PostLog(team_id=ws.team_id, post_date=today))
                db.commit()

            except Exception as e:
                print(f"[scheduler] post failed for team {ws.team_id}: {e}")

__all__ = ["run_once", "should_post_now"]
