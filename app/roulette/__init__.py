from flask import Blueprint

roulette_bp = Blueprint(
    "roulette",
    __name__,
    template_folder="../templates",
    static_folder="../static",
)

from . import routes  # noqa: E402,F401
