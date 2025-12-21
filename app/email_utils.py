import os
from flask import current_app
from postmarker.core import PostmarkClient


def send_email(to_email: str, subject: str, html_body: str, text_body: str | None = None) -> None:
    """
    Minimal email sender using Postmark when configured.
    Falls back to logging when credentials are missing.
    """
    token = os.getenv("POSTMARK_SERVER_TOKEN")
    from_email = os.getenv("EMAIL_FROM") or os.getenv("MAGIC_LINK_FROM")
    if not token or not from_email:
        # Avoid raising in production; just log and continue
        try:
            current_app.logger.info("Email send skipped (missing Postmark config) to=%s subject=%s", to_email, subject)
        except Exception:
            pass
        return
    client = PostmarkClient(server_token=token)
    try:
        client.emails.send(
            From=from_email,
            To=to_email,
            Subject=subject,
            HtmlBody=html_body,
            TextBody=text_body or "",
            MessageStream="outbound",
        )
    except Exception as exc:
        try:
            current_app.logger.error("Email send failed: %s", exc)
        except Exception:
            pass
