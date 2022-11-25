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

    def f():
        from src.data_structures.approximate_nearest_neighbor import (
            instantiate,
            read_data,
        )

        print("READING DATA FOR ANN INDEX, will only run this once")
        read_data()
        print("INSTANTIATING ANN INDEX")
        instantiate(0.9)
        print("INSTANTIATED")

        # code to generate collaborative filtering embeddings
        from src.recommendation_system.recommendation_flow.utils.cf_task import (
                generate_cf_embedding
        )

        print("generating cf embedding")
        generate_cf_embedding()

        from src.recommendation_system.recommendation_flow.utils.score_task import (
            add_image_scores
        )

        # code to insert image quality
        score_file = "/usr/src/app/image_quality.csv"
        if os.path.isfile(score_file):
            print("SCORE TASK: adding image quality scores to the table")
            add_image_scores(score_file)
        else:
            print("SCORE_TASK: score file not found. Skipping")

    app.before_first_request(f)
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

    return app
