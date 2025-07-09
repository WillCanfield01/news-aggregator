# 📰 The Roundup – News Aggregator

A Flask-based AI-powered news aggregator that pulls from global and local RSS feeds, summarizes content with GPT-4.1-mini, and detects political bias. Built for speed, clarity, and neutrality.

---

## 🚀 Features

- Aggregates articles from top RSS feeds
- Summarizes using OpenAI GPT-4.1-mini
- Classifies topics (Tech, Politics, Sports, etc.)
- Detects political bias (0–100 scale)
- Personalized local news by ZIP code
- User accounts with saved articles
- Email confirmation with Postmark
- Fast background refresh system

---

## 🛠️ Tech Stack

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **ML/NLP**: Transformers (CardiffNLP), OpenAI GPT
- **Async**: `aiohttp`, `asyncio`
- **Frontend**: Jinja2 templates, CSS
- **Database**: PostgreSQL (Render/Heroku compatible)
- **Email**: Postmark
- **Location**: `pgeocode` (ZIP → City/State)

---

## 🧪 Setup Instructions

1. **Clone the project**

```bash
git clone https://github.com/willcanfield/news-aggregator.git
cd news-aggregator