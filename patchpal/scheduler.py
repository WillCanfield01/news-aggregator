# patchpal/scheduler.py  (replace your file with this version)
from datetime import datetime
import pytz
from slack_sdk import WebClient
from .storage import SessionLocal, Workspace, PostLog
from .selector import topN_today, render_item_text
from .billing import is_active, ensure_trial, dm_trial_or_checkout

def should_post_now(tz_str: str | None, hhmm: str | None) -> bool:
    try:
        tz = pytz.timezone(tz_str or "America/Boise")
    except Exception:
        tz = pytz.timezone("America/Boise")
    now = datetime.now(tz)
    try:
        h, m = map(int, (hhmm or "09:00").split(":"))
    except Exception:
        h, m = 9, 0
    return now.hour == h and now.minute == m and now.weekday() < 5  # Mon–Fri

def run_once(client: WebClient) -> None:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    with SessionLocal() as db:
        teams = db.query(Workspace).all()
        for ws in teams:
            # ensure every workspace has a trial start date
            ensure_trial(ws, db)

            # check schedule gate
            if not (ws.post_channel and should_post_now(ws.tz, ws.post_time)):
                continue

            # one post per day
            if db.query(PostLog).filter_by(team_id=ws.team_id, post_date=today).first():
                continue

            # 🔒 billing: block & DM if not active
            if not is_active(ws=ws):
                dm_trial_or_checkout(client, ws)
                continue

            # build items
            items = topN_today(5)
            if not items:
                continue

            try:
                # header
                hdr = client.chat_postMessage(
                    channel=ws.post_channel,
                    text="Today’s Top 5 Patches / CVEs",
                    blocks=[{
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Today’s Top 5 Patches / CVEs", "emoji": True},
                    }],
                )
                parent_ts = hdr["ts"]

                # thread items
                tone = getattr(ws, "tone", "simple") or "simple"
                for i, it in enumerate(items, 1):
                    txt = render_item_text(it, i, tone)
                    client.chat_postMessage(
                        channel=ws.post_channel,
                        thread_ts=parent_ts,
                        text=f"{i})",
                        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": txt}}],
                    )

                # footer
                client.chat_postMessage(
                    channel=ws.post_channel,
                    thread_ts=parent_ts,
                    text="",
                    blocks=[{"type": "context", "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal · by The Real Roundup"}]}],
                )

                # mark success
                db.add(PostLog(team_id=ws.team_id, post_date=today))
                db.commit()

            except Exception as e:
                print(f"[scheduler] post failed for team {ws.team_id}: {e}")
