import os

# Bind to the port dynamically
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = 1  # Use a single worker to limit memory usage
timeout = 120  # Increase timeout if necessary