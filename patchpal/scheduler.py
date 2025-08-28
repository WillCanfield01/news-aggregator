# patchpal/scheduler.py
from __future__ import annotations
import time, random
from datetime import datetime
import pytz
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .install_store import get_bot_token
from .storage import SessionLocal, Workspace, PostLog
from .selector import topN_today, render_item_text, mark_posted

from .billing import (
    is_active,
    ensure_trial,
    dm_trial_or_checkout,
    dm_trial_ending_soon,
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

# --- small utils -------------------------------------------------------------

_LINK_RE = __import__("re").compile(r"<([^>|]+)\|([^>]+)>")
_TAG_RE  = __import__("re").compile(r"<[@#!][^>]+>")
_FMT_RE  = __import__("re").compile(r"[*_`~]")

def _fallback_text(md: str, limit: int = 300) -> str:
    if not isinstance(md, str):
        md = str(md or "")
    s = _LINK_RE.sub(r"\2", md)
    s = _TAG_RE.sub("", s)
    s = _FMT_RE.sub("", s)
    s = " ".join(s.split())
    return (s[:limit].rstrip() + "…") if len(s) > limit else s

def _post_with_retry(client: WebClient, **kwargs):
    """chat.postMessage with Slack 429 handling & jittered backoff."""
    attempts = 5
    for i in range(attempts):
        try:
            return client.chat_postMessage(**kwargs)
        except SlackApiError as e:
            status = getattr(e.response, "status_code", None)
            err = (e.response.get("error") if hasattr(e, "response") else None) or ""
            if status == 429 or err in {"ratelimited", "rate_limited"}:
                retry = int(e.response.headers.get("Retry-After", "1")) if hasattr(e, "response") else 1
                # small jitter that grows with attempts
                time.sleep(retry + random.uniform(0, 0.5 * (i + 1)))
                continue
            raise

# --- main tick ---------------------------------------------------------------

def run_once() -> None:
    """One tick of the scheduler. Safe to call every minute."""
    today = datetime.utcnow().strftime("%Y-%m-%d")

    with SessionLocal() as db:
        teams = db.query(Workspace).all()

        for ws in teams:
            # 0) Ensure every workspace has a trial
            ensure_trial(ws, db)

            # 0.1) Pre-expiry heads-up (non-blocking)
            try:
                dm_trial_ending_soon(None, ws)  # we’ll DM with a per-team client later if needed
            except Exception:
                pass

            # 1) Only proceed at the configured time and if a channel is set
            if not (ws.post_channel and should_post_now(ws.tz, ws.post_time)):
                continue

            # 1.1) One post per day per workspace (idempotent under concurrency)
            if db.query(PostLog).filter_by(team_id=ws.team_id, post_date=today).first():
                continue

            # 2) Billing gate
            if not is_active(ws=ws):
                try:
                    # Use a client if we have one; else the DM helper will no-op
                    tok = get_bot_token(ws.team_id)
                    if tok:
                        dm_trial_or_checkout(WebClient(token=tok), ws)
                except Exception:
                    pass
                continue

            # 3) Build a per-workspace Slack client
            token = get_bot_token(ws.team_id)
            if not token:
                print(f"[scheduler] no install/token for team {ws.team_id}")
                continue
            client = WebClient(token=token)

            # 3.1) Sanity: token belongs to this team
            try:
                auth = client.auth_test()
                if auth.get("team_id") != ws.team_id:
                    # Wrong/expired install
                    print(f"[scheduler] auth mismatch for team {ws.team_id}; skipping (reinstall needed)")
                    try:
                        # Try a gentle nudge if we know the contact
                        if ws.contact_user_id:
                            im = client.conversations_open(users=ws.contact_user_id)
                            client.chat_postMessage(
                                channel=im["channel"]["id"],
                                text="My authorization looks out of date for this workspace. Please reinstall me and try again."
                            )
                    except Exception:
                        pass
                    continue
            except SlackApiError as e:
                print(f"[scheduler] auth.test failed for team {ws.team_id}: {e.response.get('error')}")
                continue

            # 3.2) Channel existence + membership
            try:
                info = client.conversations_info(channel=ws.post_channel)
            except SlackApiError as e:
                err = e.response.get("error")
                print(f"[scheduler] conversations.info failed for {ws.post_channel}: {err}")
                continue

            ch = info.get("channel") or {}
            if ch.get("is_archived"):
                print(f"[scheduler] channel archived: {ws.post_channel}")
                continue

            is_private = ch.get("is_private")
            is_member  = ch.get("is_member")

            if not is_member:
                if is_private:
                    # cannot auto-join private channels
                    print(f"[scheduler] not a member of private channel {ws.post_channel} — invite the app")
                    # optional gentle DM
                    try:
                        if ws.contact_user_id:
                            im = client.conversations_open(users=ws.contact_user_id)
                            client.chat_postMessage(
                                channel=im["channel"]["id"],
                                text=f"I’m not in the private channel <#{ws.post_channel}>. Invite me with `/invite @PatchPal`."
                            )
                    except Exception:
                        pass
                    continue
                # public: try to join
                try:
                    client.conversations_join(channel=ws.post_channel)
                except SlackApiError as e:
                    print(f"[scheduler] join failed for {ws.post_channel}: {e.response.get('error')}")
                    continue

            # 4) Fetch items
            items = topN_today(5, ws=ws, team_id=ws.team_id)
            if not items:
                continue

            # 5) Post header
            try:
                hdr = _post_with_retry(
                    client,
                    channel=ws.post_channel,
                    text="Today’s Top 5 Patches / CVEs",
                    blocks=[{
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Today’s Top 5 Patches / CVEs", "emoji": True},
                    }],
                )
            except SlackApiError as e:
                print(f"[scheduler] header post failed for team {ws.team_id}: {e.response.get('error')}")
                continue

            parent_ts = hdr["ts"]

            # 6) Post each item as a thread reply (with fallback)
            tone = (ws.tone or "simple")
            for i, it in enumerate(items, 1):
                txt = render_item_text(it, i, tone) or f"{i}) (no details)"
                try:
                    _post_with_retry(
                        client,
                        channel=ws.post_channel,
                        thread_ts=parent_ts,
                        text=_fallback_text(txt, 300),
                        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": txt}}],
                    )
                except SlackApiError as e:
                    print(f"[scheduler] item {i} failed for team {ws.team_id}: {e.response.get('error')}")

            # 7) Footer context (non-critical)
            try:
                _post_with_retry(
                    client,
                    channel=ws.post_channel,
                    thread_ts=parent_ts,
                    text="",
                    blocks=[{
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": "Auto-posted by PatchPal · by The Real Roundup"}],
                    }],
                )
            except Exception:
                pass

            # 8) Mark success and remember items for de-dupe
            try:
                mark_posted(ws.team_id, items)
            except Exception:
                pass

            db.add(PostLog(team_id=ws.team_id, post_date=today))
            db.commit()
