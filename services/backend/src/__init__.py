import os

from flask import Flask
from flask_admin import Admin
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix

# instantiate the extensions
db = SQLAlchemy()
cors = CORS()
bcrypt = Bcrypt()
admin = Admin(template_mode="bootstrap3")


def create_app(script_info=None):
    # instantiate the app
    app = Flask(__name__)

    def before_first_request():
        from src.data_structures.approximate_nearest_neighbor import (
            instantiate,
            read_data,
        )
        from src.data_structures.approximate_nearest_neighbor.two_tower_ann import (
            instantiate_indexes,
        )

        print("INSTANTIATING ALL TEAMS ANNs")
        instantiate_indexes()
        print("INSTANTIATED")

        print("READING DATA FOR ANN INDEX, will only run this once")
        read_data()
        print("INSTANTIATING ANN INDEX")
        instantiate(0.9)
        print("INSTANTIATED")

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    # set config
    app_settings = os.getenv("APP_SETTINGS")
    app.config.from_object(app_settings)

    # set up extensions
    db.init_app(app)
    cors.init_app(app, resources={r"*": {"origins": "*"}})
    bcrypt.init_app(app)
    if os.getenv("FLASK_ENV") == "development":
        admin.init_app(app)

    # register api
    from src.api import api

    api.init_app(app)

    # shell context for flask cli
    @app.shell_context_processor
    def ctx():
        return {"app": app, "db": db}

    with app.app_context():
        db.create_all() # only create tables if they don't exist
        before_first_request()
    return app
