import threading
import time
import asyncio
import re
from app.utils.feed_utils import (
    preload_articles_batched,
    manual_refresh_articles,
    RSS_FEED_BATCHES,
)
from app.utils.geo_utils import fetch_google_local_feed
from app.models import User
from sqlalchemy.orm import sessionmaker
from app import db

def start_periodic_refresh(app, interval=600):
    def run():
        with app.app_context():
            while True:
                for batch in RSS_FEED_BATCHES:
                    preload_articles_batched(batch)
                time.sleep(interval)
    threading.Thread(target=run, daemon=True).start()

def start_periodic_local_refresh(app, local_articles_cache, interval=900):
    async def refresh_loop():
        with app.app_context():
            Session = sessionmaker(bind=db.engine)
            while True:
                session = Session()
                try:
                    users = session.query(User).filter(User.zipcode.isnot(None)).all()
                    # ... rest of your code ...
                finally:
                    session.close()
                # ... gather tasks, await, sleep, etc. ...

    def run():
        asyncio.run(refresh_loop())
    threading.Thread(target=run, daemon=True).start()

async def refresh_zip_feed(zipcode, local_articles_cache):
    articles = await fetch_google_local_feed(zipcode)
    local_articles_cache[zipcode] = articles