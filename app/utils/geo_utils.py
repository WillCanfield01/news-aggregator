from urllib.parse import quote_plus
from app.utils.geo_utils import make_local_news_query
import aiohttp  # use if you want true async for feed fetching
import feedparser

async def fetch_single_feed(url, limit=50):
    # Async RSS fetch (aiohttp + feedparser)
    articles = []
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
            async with session.get(url, timeout=20) as response:
                raw_data = await response.read()
        feed = feedparser.parse(raw_data)
        # ... (article parsing logic here)
        # Truncate to 'limit' and return
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return articles[:limit]

async def fetch_google_local_feed(zipcode: str, limit: int = 50):
    query = make_local_news_query(zipcode)
    if not query:
        print(f"⚠️ Could not resolve ZIP {zipcode} to city/state")
        return []

    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    print(f"📡 Fetching Google News RSS for: {query}")
    return await fetch_single_feed(url, limit=limit)