# patchpal/commands.py
import os
import re
from datetime import datetime
from slack_bolt import App
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from .storage import SessionLocal, Workspace
from .billing import ensure_trial

# Where your PatchPal legal page lives
LEGAL_URL = (os.getenv("APP_BASE_URL", "").rstrip("/") or "https://your-patchpal-host") + "/legal"

HELP = (
    "*PatchPal commands:*\n"
    "• `/patchpal set-channel here`  (or mention a channel)\n"
    "• `/patchpal set-time 09:00`  (HH:MM 24h)\n"
    "• `/patchpal set-tone simple|detailed`\n"
    "• `/patchpal set-stack windows,ms365,chrome,adobe,zoom,macos,ios,android,linux,edge,firefox,openssl,nginx,apache,postgres,mysql,sqlserver,aws,azure,gcp,cisco,fortinet,vmware`\n"
    "• `/patchpal mode universal|stack`\n"
    "• `/patchpal upgrade`  (open Stripe Checkout)\n"
    "• `/patchpal billing`  (plan, next renewal, manage link)\n"
    "• `/patchpal status`\n"
    "• `/patchpal post-now`  (test immediately)\n"
    f"• *Legal:* <{LEGAL_URL}|Terms & Privacy>\n"
)

# Regex helpers
TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")
MENTION_RE = re.compile(r"<#([A-Z0-9]+)\|[^>]+>", re.I)   # <#C123|name>
RAW_ID_RE  = re.compile(r"\b[CG][A-Z0-9]{8,}\b")         # C123... / G123...
HASH_NAME_RE = re.compile(r"#([a-z0-9._-]+)", re.I)      # #channel-name

# Allowed stack tokens
ALLOWED_TOKENS = {
    "windows","macos","linux","ios","android",
    "chrome","edge","firefox",
    "ms365","adobe","zoom",
    "openssl","nginx","apache",
    "postgres","mysql","sqlserver",
    "aws","azure","gcp",
    "cisco","fortinet","vmware",
}

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

def _normalize_stack(text: str):
    """Parse comma-separated tokens; return (accepted, rejected)."""
    toks = [t.strip().lower() for t in (text or "").split(",") if t.strip()]
    accepted, rejected = [], []
    for t in toks:
        if t in ALLOWED_TOKENS and t not in accepted:
            accepted.append(t)
        elif t and t not in rejected:
            rejected.append(t)
    return accepted, rejected

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
                    ensure_trial(ws, db)

                if user_id and ws.contact_user_id != user_id:
                    ws.contact_user_id = user_id
                    db.commit()

                # --- upgrade ---
                if text.startswith("upgrade"):
                    from .billing import checkout_url
                    url = checkout_url(team_id)
                    respond(f"Upgrade to *PatchPal Pro* ($9/workspace/mo): <{url}|Open Stripe Checkout>")
                    return

                # --- billing ---
                if text.startswith("billing"):
                    from .billing import portal_url, get_next_renewal, checkout_url
                    plan = ws.plan or "trial"
                    if plan == "pro" and ws.subscription_id:
                        nxt = get_next_renewal(ws)
                        nxt_txt = nxt.date().isoformat() if nxt else "n/a"
                        purl = portal_url(ws) or "https://billing.stripe.com/"
                        respond(
                            f"*Plan:* pro  |  *Next renewal:* {nxt_txt}\n"
                            f"*Manage billing:* <{purl}|Open Stripe Customer Portal>\n"
                            f"*Legal:* <{LEGAL_URL}|Terms & Privacy>"
                        )
                    else:
                        if ws.trial_ends_at:
                            days_left = max((ws.trial_ends_at.date() - datetime.utcnow().date()).days, 0)
                        else:
                            days_left = "n/a"
                        url = checkout_url(team_id)
                        respond(
                            f"*Plan:* trial  |  *Days left:* {days_left}\n"
                            f"*Upgrade:* <{url}|Open Stripe Checkout>\n"
                            f"*Legal:* <{LEGAL_URL}|Terms & Privacy>"
                        )
                    return

                # --- set-channel ---
                if text.startswith("set-channel"):
                    cid = _resolve_channel_id(text, body, client, logger)
                    if not cid:
                        respond(
                            "Please *mention* the channel (so Slack sends `<#C…|name>`) or run "
                            "`/patchpal set-channel here` in the target channel.\n"
                            "_Tip: enable “Escape channels, users, and links sent to your app.” in your Slash Command._\n"
                            f"*Legal:* <{LEGAL_URL}|Terms & Privacy>"
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

                # --- set-stack ---
                if text.startswith("set-stack"):
                    arg = text[len("set-stack"):].strip().lstrip("= :")
                    accepted, rejected = _normalize_stack(arg)
                    ws.stack_tokens = ",".join(accepted) if accepted else None
                    db.commit()
                    msg = f"Stack set to: `{ws.stack_tokens or 'none'}`. "
                    if rejected:
                        msg += f"(ignored unknown: {', '.join(rejected)}) "
                    msg += "Use `/patchpal mode stack` to tailor posts, or stay in `universal`."
                    respond(msg + f"\n*Legal:* <{LEGAL_URL}|Terms & Privacy>")
                    return

                # --- mode ---
                if text.startswith("mode"):
                    parts = text.split()
                    mode = (parts[-1] if len(parts) >= 2 else "").lower()
                    if mode not in ("universal", "stack"):
                        respond("Use `/patchpal mode universal` or `/patchpal mode stack`.")
                        return
                    ws.stack_mode = mode
                    db.commit()
                    respond(f"Relevance mode set to *{ws.stack_mode}*.")
                    return

                # --- status ---
                if text.startswith("status"):
                    ch = f"<#{ws.post_channel}>" if ws.post_channel else "_not set_"
                    trial = f"{(ws.trial_ends_at.isoformat()[:10] if ws.trial_ends_at else 'n/a')}"
                    plan = ws.plan or "trial"
                    stack = ws.stack_tokens or "none"
                    respond(
                        f"Channel: {ch}  |  Time: {ws.post_time}  |  Tone: *{ws.tone or 'simple'}*  "
                        f"|  Plan: *{plan}*  |  Trial ends: *{trial}*  |  TZ: {ws.tz}\n"
                        f"Relevance: *{ws.stack_mode or 'universal'}*  |  Stack: `{stack}`\n"
                        f"*Legal:* <{LEGAL_URL}|Terms & Privacy>"
                    )
                    return

                # --- post-now (threaded) ---
                if text.startswith("post-now"):
                    if not ws.post_channel:
                        respond("Set a channel first: `*/patchpal set-channel here*` in the target channel.")
                        return

                    # Import here to avoid circulars and keep cold start fast
                    from .selector import topN_today, render_item_text

                    items = topN_today(5, ws=ws)  # always 5, and pass workspace for relevance
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

                    # 2) Post each item as a thread reply
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
