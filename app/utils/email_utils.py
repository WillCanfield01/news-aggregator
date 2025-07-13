import os
from itsdangerous import URLSafeTimedSerializer
from postmarker.core import PostmarkClient

# Ensure environment variables are present and valid
POSTMARK_SERVER_TOKEN = os.getenv("POSTMARK_SERVER_TOKEN")
EMAIL_FROM = os.getenv("EMAIL_FROM")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-key")

if not POSTMARK_SERVER_TOKEN:
    raise RuntimeError("POSTMARK_SERVER_TOKEN environment variable not set.")
if not EMAIL_FROM:
    raise RuntimeError("EMAIL_FROM environment variable not set.")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable not set.")

postmark = PostmarkClient(server_token=POSTMARK_SERVER_TOKEN)

def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    return serializer.dumps(email, salt="email-confirmation-salt")

def confirm_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    try:
        return serializer.loads(token, salt="email-confirmation-salt", max_age=expiration)
    except Exception as e:
        print("Error confirming token:", e)
        return False

def send_confirmation_email(email, username, token):
    confirm_link = f"https://therealroundup.com/confirm/{token}"
    try:
        postmark.emails.send(
            From=EMAIL_FROM,
            To=email,
            Subject='Confirm Your Email – The Roundup',
            HtmlBody=f'''
                <p>Hi {username},</p>
                <p>Please confirm your email by clicking the link below:</p>
                <p><a href="{confirm_link}">Confirm your email</a></p>
            ''',
            MessageStream="outbound"
        )
        print(f"Confirmation email sent to {email}")
        return True
    except Exception as e:
        print("Error sending confirmation email:", e)
        return False