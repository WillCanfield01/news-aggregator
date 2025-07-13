from urllib.parse import quote_plus
import aiohttp  # use if you want true async for feed fetching
import feedparser
import pgeocode
import pandas as pd

nomi = pgeocode.Nominatim('us')

def get_city_state_from_zip(zipcode):
    """Looks up city and state from a US ZIP using pgeocode."""
    zipcode = str(zipcode).zfill(5)
    info = nomi.query_postal_code(zipcode)
    if pd.notna(info.place_name) and pd.notna(info.state_name):
        return info.place_name, info.state_name
    return None, None

def make_local_news_query(zipcode):
    """Generate a local news query string based on ZIP code."""
    city, state = get_city_state_from_zip(zipcode)
    if city and state:
        return f'"{city}, {state}" local news'
    else:
        return None

async def fetch_single_feed(url, limit=50):
    # Async RSS fetch (aiohttp + feedparser)
    articles = []
    try:
        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
            async with session.get(url, timeout=20) as response:
                raw_data = await response.read()
        feed = feedparser.parse(raw_data)
        # Parse articles from feed
        for entry in feed.entries[:limit]:
            article = {
                "title": entry.get("title"),
                "link": entry.get("link"),
                "published": entry.get("published"),
                "summary": entry.get("summary"),
                "source": feed.feed.get("title"),
            }
            articles.append(article)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []
    return articles

async def fetch_google_local_feed(zipcode: str, limit: int = 50):
    query = make_local_news_query(zipcode)
    if not query:
        print(f"⚠️ Could not resolve ZIP {zipcode} to city/state")
        return []

    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    print(f"📡 Fetching Google News RSS for: {query}")
    return await fetch_single_feed(url, limit=limit)