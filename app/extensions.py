# app/extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Single, shared instances used across the app
db = SQLAlchemy()
login_manager = LoginManager()
