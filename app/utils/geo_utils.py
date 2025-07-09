import re
import pgeocode
import pandas as pd
from urllib.parse import quote_plus
from app.utils.feed_utils import fetch_single_feed

nomi = pgeocode.Nominatim('us')

def get_city_state_from_zip(zipcode):
    info = nomi.query_postal_code(zipcode)
    if pd.notna(info.place_name) and pd.notna(info.state_name):
        return info.place_name, info.state_name
    return None, None

async def fetch_google_local_feed(zipcode, limit=50):
    city, state = get_city_state_from_zip(zipcode)
    if not city or not state:
        print(f"Could not resolve ZIP {zipcode}")
        return []
    query = f"{city} {state} local news"
    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    return await fetch_single_feed(url, limit=limit)