import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"  # Use dynamic port from Render
workers = 1
timeout = 120
