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

def start_periodic_refresh(app, interval=480):
    """Background batch refresh for main articles (uses batch cycling)"""
    def run():
        with app.app_context():
            batch = 0
            while True:
                preload_articles_batched(RSS_FEED_BATCHES[batch])
                batch = (batch + 1) % len(RSS_FEED_BATCHES)
                time.sleep(interval)
    threading.Thread(target=run, daemon=True).start()

def start_periodic_local_refresh(app, local_articles_cache, interval=900):
    """Background refresh for local (ZIP) feeds using Google News RSS"""
    async def refresh_loop():
        with app.app_context():
            from app import db  # Import here to avoid circular import
            while True:
                users = User.query.filter(User.zipcode.isnot(None)).all()
                seen_zips = set()
                tasks = []
                for user in users:
                    zip_code = user.zipcode.strip()
                    if re.match(r"^\d{5}$", zip_code) and zip_code not in seen_zips:
                        seen_zips.add(zip_code)
                        tasks.append(refresh_zip_feed(zip_code, local_articles_cache))
                await asyncio.gather(*tasks)
                await asyncio.sleep(interval)

    def run():
        asyncio.run(refresh_loop())

    threading.Thread(target=run, daemon=True).start()

async def refresh_zip_feed(zipcode, local_articles_cache):
    articles = await fetch_google_local_feed(zipcode)
    local_articles_cache[zipcode] = articles