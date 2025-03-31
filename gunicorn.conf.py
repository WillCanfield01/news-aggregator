# gunicorn.conf.py
import os

loglevel = 'debug'  # Detailed logs
accesslog = '-'  # Logs to stdout
errorlog = '-'  # Logs to stderr

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 1
timeout = 120
