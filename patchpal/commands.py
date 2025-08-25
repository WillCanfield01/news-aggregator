# patchpal/commands.py
import os
import re
from datetime import datetime
from slack_bolt import App
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from .storage import SessionLocal, Workspace
from .billing import ensure_trial

LEGAL_URL = (os.getenv("APP_BASE_URL", "").rstrip("/") or "https://your-patchpal-host") + "/legal"

HELP = (
    "*PatchPal commands:*\n"
    "• `/patchpal set-channel here`  (or mention a channel)\n"
    "• `/patchpal set-time 09:00`  (HH:MM 24h)\n"
    "• `/patchpal set-tone simple|detailed`\n"
    "• `/patchpal set-stack windows, ms365, chrome`  (choose techs)\n"
    "• `/patchpal mode universal|stack`  (universal = broad; stack = tailored)\n"
    "• `/patchpal stacks`  (see options & examples)\n"
    "• `/patchpal show-stack`  (see what’s set)\n"
    "• `/patchpal preset office-it|mac-shop|cloud-dev|network`  (quick setup)\n"
    "• `/patchpal upgrade`  (open Stripe Checkout)\n"
    "• `/patchpal billing`  (plan, next renewal, manage link)\n"
    "• `/patchpal status`\n"
    "• `/patchpal post-now`  (test immediately)\n"
    f"• *Legal:* <{LEGAL_URL}|Terms & Privacy>\n"
)

# ── Channel parsing helpers ───────────────────────────────────────────────────
TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")
MENTION_RE = re.compile(r"<#([A-Z0-9]+)\|[^>]+>", re.I)  # <#C123|name>
RAW_ID_RE  = re.compile(r"\b[CG][A-Z0-9]{8,}\b")        # C123... or G123...
HASH_NAME_RE = re.compile(r"#([a-z0-9._-]+)", re.I)     # #channel-name

def _resolve_channel_id(text: str, body: dict, client, logger):
    t = (text or "").strip()
    if t.lower().endswith("here"):
        return body.get("channel_id")
    m = MENTION_RE.search(t)
    if m:
        return m.group(1)
    m = RAW_ID_RE.search(t)
    if m:
        return m.group(0)
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
        logger.info(f"conversations.list failed or missing scopes: {e}")
    return None

# ── Stack options (grouped) ───────────────────────────────────────────────────
STACK_CATALOG = {
    "OS & Browsers": ["windows", "macos", "linux", "ios", "android", "chrome", "edge", "firefox"],
    "Cloud & Email": ["ms365", "aws", "azure", "gcp", "zoom"],
    "Apps & Libs":   ["adobe", "openssl"],
    "Servers & DBs": ["nginx", "apache", "postgres", "mysql", "sqlserver"],
    "Network/Infra": ["cisco", "fortinet", "vmware"],
}
ALLOWED_TOKENS = {t for group in STACK_CATALOG.values() for t in group}

# friendly synonyms
SYNONYMS = {
    "office": "ms365",
    "o365": "ms365",
    "m365": "ms365",
    "sql": "sqlserver",
    "mac": "macos",
    "win": "windows",
    "chromeos": "chrome",  # if someone types it
}

PRESETS = {
    "office-it": ["windows", "ms365", "chrome", "zoom"],
    "mac-shop":  ["macos", "ios", "chrome", "adobe"],
    "cloud-dev": ["linux", "aws", "azure", "gcp", "nginx", "postgres"],
    "network":   ["linux", "cisco", "fortinet", "vmware"],
}

def _parse_stack_args(arg: str):
    # accept commas OR spaces; normalize synonyms
    raw = [x.strip().lower() for x in re.split(r"[, ]+", arg or "") if x.strip()]
    cleaned, rejected = [], []
    for tok in raw:
        tok = SYNONYMS.get(tok, tok)
        if tok in ALLOWED_TOKENS and tok not in cleaned:
            cleaned.append(tok)
        elif tok not in ALLOWED_TOKENS:
            rejected.append(tok)
    return cleaned, rejected

def _catalog_text():
    # compact, readable catalog + examples
    lines = ["*Choose your stack* (comma-separated), e.g. `windows, ms365, chrome`"]
    for title, items in STACK_CATALOG.items():
        lines.append(f"• *{title}:* " + ", ".join(items))
    lines.append(
        "\nExamples:\n"
        "• Windows shop: `windows, ms365, chrome`\n"
        "• Mac shop: `macos, ios, chrome, adobe`\n"
        "• Cloud/dev: `linux, aws, azure, gcp, nginx, postgres`\n"
        "Tip: Switch targeting with `/patchpal mode stack` (tailored) or `universal` (broad)."
    )
    return "\n".join(lines)

# ── Command router ────────────────────────────────────────────────────────────
def register_commands(app: App):
    @app.command("/patchpal")
    def patchpal_cmd(ack, body, respond, client, logger):
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

                # ---------- quick info commands ----------
                if text.startswith("stacks"):
                    respond(_catalog_text())
                    return

                if text.startswith("show-stack"):
                    stack = ws.stack_tokens or "none"
                    mode = ws.stack_mode or "universal"
                    respond(f"*Mode:* {mode}  |  *Stack:* `{stack}`\n"
                            "Change with `/patchpal set-stack …` or `/patchpal mode stack`.")
                    return

                if text.startswith("preset"):
                    key = (text.split(maxsplit=1)[1].strip().lower() if len(text.split()) > 1 else "")
                    if key not in PRESETS:
                        respond("Presets: `office-it`, `mac-shop`, `cloud-dev`, `network`.\n"
                                "Use like: `/patchpal preset office-it`.")
                        return
                    ws.stack_tokens = ",".join(PRESETS[key])
                    ws.stack_mode = "stack"
                    db.commit()
                    respond(f"Preset *{key}* applied. Mode set to *stack*. "
                            f"Current stack: `{ws.stack_tokens}`.")
                    return

                # ---------- billing ----------
                if text.startswith("upgrade"):
                    from .billing import checkout_url
                    url = checkout_url(team_id)
                    respond(f"Upgrade to *PatchPal Pro* ($9/workspace/mo): <{url}|Open Stripe Checkout>")
                    return

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
                        days_left = max((ws.trial_ends_at.date() - datetime.utcnow().date()).days, 0) if ws.trial_ends_at else "n/a"
                        url = checkout_url(team_id)
                        respond(
                            f"*Plan:* trial  |  *Days left:* {days_left}\n"
                            f"*Upgrade:* <{url}|Open Stripe Checkout>\n"
                            f"*Legal:* <{LEGAL_URL}|Terms & Privacy>"
                        )
                    return

                # ---------- settings ----------
                if text.startswith("set-channel"):
                    cid = _resolve_channel_id(text, body, client, logger)
                    if not cid:
                        respond(
                            "Please *mention* the channel or run `/patchpal set-channel here` in the target channel.\n"
                            "_Tip: enable “Escape channels, users, and links sent to your app.” in your Slash Command._\n"
                            f"*Legal:* <{LEGAL_URL}|Terms & Privacy>"
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
                        respond("Use HH:MM 24h, e.g., `09:00`.")
                        return
                    h, m = hhmm.split(":")
                    ws.post_time = f"{int(h):02d}:{int(m):02d}"
                    db.commit()
                    respond(f"Time set to {ws.post_time}.")
                    return

                if text.startswith("set-tone"):
                    tone = (text.split(maxsplit=1)[1].strip().lower() if len(text.split()) > 1 else "")
                    if tone not in ("simple", "detailed"):
                        respond("Use `simple` or `detailed`, e.g., `/patchpal set-tone simple`.")
                        return
                    ws.tone = tone
                    db.commit()
                    respond(f"Tone set to *{ws.tone}*.")
                    return

                if text.startswith("mode"):
                    mode = (text.split(maxsplit=1)[1].strip().lower() if len(text.split()) > 1 else "")
                    if mode not in ("universal", "stack"):
                        respond("Use `/patchpal mode universal` or `/patchpal mode stack`.")
                        return
                    ws.stack_mode = mode
                    db.commit()
                    respond(f"Relevance mode set to *{ws.stack_mode}*.")
                    return

                if text.startswith("set-stack"):
                    arg = text.split(" ", 1)[1] if " " in text else ""
                    chosen, rejected = _parse_stack_args(arg)
                    ws.stack_tokens = ",".join(chosen) if chosen else None
                    db.commit()
                    msg = f"Stack set to: `{ws.stack_tokens or 'none'}`."
                    if rejected:
                        msg += f" (ignored unknown: {', '.join(rejected)})"
                    msg += " Run `/patchpal mode stack` for tailored posts, or stay in `universal`."
                    respond(msg + f"\nSee options: `/patchpal stacks`  ·  *Legal:* <{LEGAL_URL}|Terms & Privacy>")
                    return

                # ---------- info & action ----------
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

                if text.startswith("post-now"):
                    if not ws.post_channel:
                        respond("Set a channel first: `*/patchpal set-channel here*` in the target channel.")
                        return
                    from .selector import topN_today, render_item_text
                    items = topN_today(5, ws=ws)
                    if not items:
                        respond("No items found right now.")
                        return
                    hdr = client.chat_postMessage(
                        channel=ws.post_channel,
                        text="Today’s Top 5 Patches / CVEs",
                        blocks=[{"type":"header","text":{"type":"plain_text","text":"Today’s Top 5 Patches / CVEs","emoji":True}}],
                    )
                    parent_ts = hdr["ts"]
                    for i, it in enumerate(items, 1):
                        txt = render_item_text(it, i, ws.tone or "simple")
                        # Belt & suspenders: ensure string & within Slack limits
                        if not isinstance(txt, str):
                            txt = str(txt or "")
                        txt = txt.strip()
                        if not txt:
                            txt = f"{i}) (no details)"
                        if len(txt) > 2900:
                            txt = txt[:2900] + "…"

                        client.chat_postMessage(
                            channel=ws.post_channel,
                            thread_ts=parent_ts,
                            text=f"{i})",  # fallback text
                            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": txt}}],
                        )
                    respond(f"Posted {len(items)} item(s) to <#{ws.post_channel}>.")
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
