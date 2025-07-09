import os
from itsdangerous import URLSafeTimedSerializer
from postmarker.core import PostmarkClient

postmark = PostmarkClient(server_token=os.getenv("POSTMARK_SERVER_TOKEN"))
SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")

def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    return serializer.dumps(email, salt="email-confirmation-salt")

def confirm_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    try:
        return serializer.loads(token, salt="email-confirmation-salt", max_age=expiration)
    except Exception:
        return False

def send_confirmation_email(email, username, token):
    confirm_link = f"https://therealroundup.com/confirm/{token}"
    postmark.emails.send(
        From=os.getenv("EMAIL_FROM"),
        To=email,
        Subject='Confirm Your Email – The Roundup',
        HtmlBody=f'''
            <p>Hi {username},</p>
            <p>Please confirm your email by clicking the link below:</p>
            <p><a href="{confirm_link}">Confirm your email</a></p>
        ''',
        MessageStream="outbound"
    )