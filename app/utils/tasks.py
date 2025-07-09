import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import current_app
from app.utils.feed_utils import fetch_feed
from app.utils.geo_utils import fetch_google_local_feed
from yourapp.models import User  # Adjust to match your project

MAX_CACHED_ARTICLES = 300
cached_articles = []
new_articles_last_refresh = []

RSS_FEED_BATCHES = [ [...], [...] ]  # Define this in app config or shared module

def preload_articles_batched(feed_list, use_ai=False):
    global cached_articles, new_articles_last_refresh
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda url: fetch_feed(url, use_ai), feed_list)
        new_articles = [a for feed in results for a in feed]
    existing_ids = {a["id"] for a in cached_articles}
    unique_new = [a for a in new_articles if a["id"] not in existing_ids]
    new_articles_last_refresh = unique_new
    cached_articles = unique_new + cached_articles
    cached_articles = cached_articles[:MAX_CACHED_ARTICLES]

def periodic_refresh(interval=480):
    def run():
        batch = 0
        while True:
            preload_articles_batched(RSS_FEED_BATCHES[batch])
            batch = (batch + 1) % len(RSS_FEED_BATCHES)
            time.sleep(interval)
    threading.Thread(target=run, daemon=True).start()

def periodic_zip_refresh(local_articles_cache, db_session, interval=900):
    async def refresh_loop():
        while True:
            users = db_session.query(User).filter(User.zipcode.isnot(None)).all()
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