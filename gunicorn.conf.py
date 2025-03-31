import os

bind = f"0.0.0.0:{os.getenv('PORT')}"  # Ensure Render assigns the correct port
workers = 1
timeout = 120
