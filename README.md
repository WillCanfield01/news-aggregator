📰 The Roundup – AI News & Community Aggregator
A Flask-based AI-powered news aggregator and article generator. Get summarized headlines from top global and local sources, view community-generated feature articles, and detect political bias—all built for speed, clarity, and privacy.

🚀 Features
Aggregates articles from top global and local RSS feeds

AI summarization using OpenAI GPT models

Automatic topic classification (Tech, Politics, Sports, etc.)

Political bias detection (0–100 scale, center = neutral)

Personalized local news by user ZIP code

User accounts with email confirmation (via Postmark)

Save your favorite articles to your account

Community feature article generator (inspired by trending topics)

Clean, fast UI with mobile-friendly templates

Automated background article refresh

Works on Render, Heroku, or any Flask-friendly server

🛠️ Tech Stack
Backend: Flask, SQLAlchemy, Flask-Login

ML/NLP: Transformers (CardiffNLP), OpenAI GPT API

Async: aiohttp, asyncio

Frontend: Jinja2 templates, CSS/Bootstrap

Database: PostgreSQL (Render/Heroku compatible)

Email: Postmark API

Location: pgeocode (ZIP → City/State)

Community Articles: Python, OpenAI, Markdown

🧪 Setup Instructions
Clone the project:

bash
Copy
Edit
git clone https://github.com/willcanfield/news-aggregator.git
cd news-aggregator
Create and activate a virtual environment:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
Install requirements:

bash
Copy
Edit
pip install -r requirements.txt
Set up your environment variables:
Create a .env file or export these variables in your shell:

ini
Copy
Edit
FLASK_ENV=development
SECRET_KEY=your_secret_key
DATABASE_URL=your_database_url  # e.g. postgres://...
OPENAI_API_KEY=sk-...
POSTMARK_SERVER_TOKEN=your_postmark_token
EMAIL_FROM=your@email.com
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
Initialize the database:

bash
Copy
Edit
flask shell
>>> from app import db, create_app
>>> app = create_app()
>>> with app.app_context():
...     db.create_all()
... 
Run the application:

bash
Copy
Edit
flask run
Or with gunicorn (for production):

bash
Copy
Edit
gunicorn "app:create_app()"
⚙️ Usage
Landing page: Summarized global and local news headlines, with bias and category filters.

User accounts: Sign up, confirm via email, and start saving your favorite articles.

Local news: Enter your ZIP code and view personalized local headlines.

Community feature articles: Generate a new, AI-powered article daily, or explore previous features.

Admin: You can manually generate a new article via /reddit-articles (or set up as a scheduled task).

👥 Community Article Generator
Generate a new feature article each day inspired by trending online questions.

All articles are AI-written and not directly attributed to any platform.

Explore all articles at /reddit-articles/articles.

🕒 Scheduling
For automated community article creation (e.g., daily at 5 PM MST), use a cron job or task scheduler:

bash
Copy
Edit
0 17 * * * cd /path/to/news-aggregator && /path/to/venv/bin/python app/reddit_articles.py
📝 License
MIT (or your chosen license).
This project is for educational and personal use—do not redistribute AI-generated news summaries for commercial purposes without proper compliance and rights.

✨ Credits
Made by Will Canfield.
OpenAI, CardiffNLP, Flask, and the open-source news community.

Enjoy your smarter, bias-aware news and article platform!