from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from config import config

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_name='default'):
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Register routes
    from app.routes import auth_routes, news_routes, local_routes, user_routes
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(news_routes.bp)
    app.register_blueprint(local_routes.bp)
    app.register_blueprint(user_routes.bp)

    return app