# pip install tweepy pytrends feedparser psycopg2-binary SQLAlchemy pandas scikit-learn python-dotenv

import os, time, random, re, datetime as dt
import pandas as pd
import feedparser
from pytrends.request import TrendReq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tweepy
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
BASE_URL = os.getenv("BASE_URL", "https://therealroundup.com")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.25"))
POSTS_PER_RUN = int(os.getenv("POSTS_PER_RUN", "1"))
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "1"))  # don’t tweet same article/trend within X hours

X_CONSUMER_KEY     = os.getenv("X_CONSUMER_KEY")
X_CONSUMER_SECRET  = os.getenv("X_CONSUMER_SECRET")
X_ACCESS_TOKEN     = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_SECRET    = os.getenv("X_ACCESS_SECRET")

HASHTAGS = ["#trending", "#news", "#culture", "#debate", "#TheRealRoundup"]

def get_google_trends(top_n=15, country="united_states"):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        df = pytrends.trending_searches(pn=country)
        return [t.strip() for t in df[0].dropna().head(top_n).tolist() if isinstance(t, str)]
    except Exception:
        return []

def get_news_rss_terms():
    feeds = [
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
    ]
    terms = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:20]:
                terms.append(e.title)
        except Exception:
            pass
    return terms

def connect_db() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def load_articles(engine: Engine) -> pd.DataFrame:
    # Adjust field names if your model differs
    sql = """
        SELECT
          id,
          title,
          filename,
          COALESCE(meta_description, '') AS meta_description,
          COALESCE(html_content, '')  AS html_content
        FROM community_article
        WHERE title IS NOT NULL AND filename IS NOT NULL
        ORDER BY date DESC
        LIMIT 500;
    """
    df = pd.read_sql(sql, engine)
    # plain-ish text body
    def strip_html(md):
        # very light cleanup
        return re.sub(r'<[^>]+>', '', md or '')
    df["summary"] = df["meta_description"].fillna("").astype(str)
    fallback = df["html_content"].fillna("").map(strip_html)
    df.loc[df["summary"].str.len() < 40, "summary"] = fallback[df["summary"].str.len() < 40]
    df["url"] = BASE_URL.rstrip("/") + f"/all-articles/articles/" + df["filename"].astype(str)
    # UTM
    df["url"] = df["url"] + "?utm_source=twitter&utm_medium=trendbot&utm_campaign=trend_hijack"
    df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()
    return df

def build_vectorizer(corpus):
    vect = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
    X = vect.fit_transform(corpus)
    return vect, X

def best_match(term, vect, X, df):
    q = vect.transform([term])
    sims = cosine_similarity(q, X)[0]
    idx = sims.argmax()
    return int(idx), float(sims[idx])

def templated_tweet(trend, row):
    templates = [
        "Everyone’s talking about {trend}. Here’s what they miss: {title}. {url}",
        "{trend} is blowing up. Quick context: {title}. {url}",
        "Hot take on {trend}: {title}. 60s summary → {url}",
        "{trend}, explained without fluff: {title}. Read: {url}",
        "If {trend} confuses you, this helps: {title}. {url}"
    ]
    t = random.choice(templates)
    txt = t.format(trend=trend, title=row["title"], url=row["url"])
    remain = 280 - len(txt) - 1
    tags = ""
    if remain > 12:
        tags = " " + " ".join(random.sample(HASHTAGS, k=min(2, len(HASHTAGS))))
    return (txt + tags)[:280]

def recently_tweeted(engine: Engine, article_id: int, trend: str) -> bool:
    sql = text("""
        SELECT 1 FROM trend_post_log
        WHERE article_id = :aid AND trend = :trend
          AND tweeted_at >= now() - INTERVAL ':cool hours'
        LIMIT 1;
    """.replace(":cool", str(COOLDOWN_HOURS)))
    with engine.begin() as conn:
        row = conn.execute(sql, {"aid": article_id, "trend": trend}).fetchone()
        return row is not None

def log_tweet(engine: Engine, article_id: int, trend: str):
    sql = text("INSERT INTO trend_post_log (trend, article_id) VALUES (:t, :aid) ON CONFLICT DO NOTHING;")
    with engine.begin() as conn:
        conn.execute(sql, {"t": trend, "aid": article_id})

def post_to_x(status_text):
    auth = tweepy.OAuth1UserHandler(
        X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET
    )
    api = tweepy.API(auth)
    api.update_status(status=status_text)

def run_once():
    engine = connect_db()
    df = load_articles(engine)
    if df.empty:
        print("No articles found.")
        return

    trends = get_google_trends(top_n=15) + get_news_rss_terms()
    trends = [t for t in trends if isinstance(t, str)]
    # dedupe while preserving order
    seen = set(); uniq_trends = []
    for t in trends:
        if t not in seen:
            uniq_trends.append(t); seen.add(t)
    if not uniq_trends:
        print("No trends found.")
        return

    vect, X = build_vectorizer(df["text"].tolist())

    candidates = []
    for term in uniq_trends:
        idx, sim = best_match(term, vect, X, df)
        if sim >= SIM_THRESHOLD:
            row = df.iloc[idx]
            # cooldown check
            if recently_tweeted(engine, int(row["id"]), term):
                continue
            tweet = templated_tweet(term, row)
            candidates.append((sim, term, int(row["id"]), tweet))

    # No good matches
    if not candidates:
        print("No candidate tweets this run.")
        return

    # Sort by similarity desc, post top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    for sim, trend, aid, tweet in candidates[:POSTS_PER_RUN]:
        try:
            post_to_x(tweet)
            log_tweet(engine, aid, trend)
            print(f"Tweeted [{trend}] → article_id {aid} (sim={sim:.3f})")
            time.sleep(2)
        except Exception as e:
            print("Post failed:", e)

if __name__ == "__main__":
    run_once()
