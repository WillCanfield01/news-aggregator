# app/extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from app.extensions import db, login_manager

db = SQLAlchemy()
login_manager = LoginManager()
