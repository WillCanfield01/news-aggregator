from app import db
from datetime import datetime

class CommunityArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    filename = db.Column(db.String(200), unique=True, nullable=False)
    title = db.Column(db.String(300), nullable=False)
    content = db.Column(db.Text, nullable=False)  # Store the markdown here
    html_content = db.Column(db.Text)  # Optional: store the rendered HTML
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta_title = db.Column(db.String(255), nullable=True)
    meta_description = db.Column(db.Text, nullable=True)
