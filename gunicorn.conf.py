import os

bind = f"0.0.0.0:{os.environ['PORT']}"  # Ensure it dynamically binds without fallback
workers = 1
timeout = 120