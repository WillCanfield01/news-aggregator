import os
from sqlalchemy import create_engine, text
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

DATABASE_URL = "postgresql://neondb_owner:npg_5psGM4LhFOfe@ep-purple-band-a6ukmj3s-pooler.us-west-2.aws.neon.tech/neondb"
# ---- DB Setup ----
engine = create_engine(DATABASE_URL)

def fetch_latest_reel_script():
    with engine.connect() as conn:
        # Replace 'reel_script' with your actual column name if different
        result = conn.execute(
            text("SELECT reel_script FROM community_article WHERE reel_script IS NOT NULL ORDER BY date DESC LIMIT 1")
        ).fetchone()
        return result[0] if result else None

# ---- Selenium Setup ----
CHROME_DRIVER_PATH = "./chromedriver_mac64/chromedriver"  # or specify full path if needed

chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--disable-notifications")

def automate_pictory_login_and_create(script_text):
    driver = webdriver.Chrome(CHROME_DRIVER_PATH, options=chrome_options)

    try:
        driver.get("https://app.pictory.ai/")
        time.sleep(3)

        # 1. Login if needed (try to use session/cookies if already logged in)
        if "login" in driver.current_url or "signin" in driver.current_url:
            print("Please login manually the first time and save your login in Chrome for future automation.")
            time.sleep(30)  # Give you time to log in; after this, automation will continue

        # 2. Go to 'Script to Video'
        # Look for the button or menu (you might need to adjust selector)
        # Click 'Script to Video'
        time.sleep(2)
        script_to_video_btn = driver.find_element(By.XPATH, "//span[contains(text(), 'Script to Video')]")
        script_to_video_btn.click()
        time.sleep(4)

        # 3. Paste the script
        textarea = driver.find_element(By.TAG_NAME, "textarea")
        textarea.clear()
        textarea.send_keys(script_text)
        time.sleep(1)

        # 4. Click 'Proceed' (adjust selector if needed)
        proceed_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Proceed')]")
        proceed_btn.click()
        time.sleep(8)

        # 5. Pick vertical format
        vertical_btn = driver.find_element(By.XPATH, "//div[contains(text(), 'Vertical (9:16)')]")
        vertical_btn.click()
        time.sleep(2)

        # 6. Click 'Proceed' again to confirm
        proceed2_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Proceed')]")
        proceed2_btn.click()
        time.sleep(10)

        print("🎬 Reel creation started! Review/export video manually if needed.")
        # Optional: Wait for video to render, then download, or use more automation if you want to grab the file.

        time.sleep(20)  # Give time for preview/load

    finally:
        driver.quit()

if __name__ == "__main__":
    script_text = fetch_latest_reel_script()
    if not script_text:
        print("No reel script found in the database.")
    else:
        print("Automating Pictory reel creation...")
        automate_pictory_login_and_create(script_text)
