from datetime import datetime
import pytz
from storage import SessionLocal, Workspace, PostLog
from selector import top3_today, as_slack_blocks
from slack_sdk import WebClient

def should_post_now(tz_str, hhmm):
    try:
        tz = pytz.timezone(tz_str or "America/Boise")
    except Exception:
        tz = pytz.timezone("America/Boise")
    now = datetime.now(tz)
    h, m = map(int, (hhmm or "09:00").split(":"))
    return now.hour == h and now.minute == m and now.weekday() < 5  # weekdays only

def run_once(client: WebClient):
    db = SessionLocal()
    teams = db.query(Workspace).all()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    for ws in teams:
        if not (ws.post_channel and should_post_now(ws.tz, ws.post_time)):
            continue
        if db.query(PostLog).filter_by(team_id=ws.team_id, post_date=today).first():
            continue
        items = top3_today()
        if not items:
            continue
        try:
            client.chat_postMessage(channel=ws.post_channel,
                                    text="Today’s Top 3 Patches / CVEs",
                                    blocks=as_slack_blocks(items))
            db.add(PostLog(team_id=ws.team_id, post_date=today)); db.commit()
        except Exception as e:
            print("post failed", ws.team_id, e)
    db.close()