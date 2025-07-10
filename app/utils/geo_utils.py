from urllib.parse import quote_plus
import aiohttp  # use if you want true async for feed fetching
import feedparser

# app/utils/geo_utils.py

def get_city_state_from_zip(zipcode):
    """Basic ZIP code to city/state lookup stub.
    In production, replace this with an actual lookup or API.
    """
    # Dummy mapping - add your own or use an API
    zip_map = {
        "90210": ("Beverly Hills", "CA"),
        "10001": ("New York", "NY"),
        "83702": ("Boise", "ID"),
        # add more as needed
    }
    result = zip_map.get(str(zipcode))
    if result:
        city, state = result
        return city, state
    else:
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