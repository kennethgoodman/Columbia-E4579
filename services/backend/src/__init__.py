import os
import traceback

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

    def before_first_request_checks():
        from sqlalchemy import func, exc
        from src.api.engagement.models import Engagement
        from src.api.content.models import Content, GeneratedContentMetadata
        from src.api.users.models import User
        for table in [Engagement, Content, GeneratedContentMetadata, User]:
            try:
                row_count = db.session.query(func.count(table.id)).scalar()
                if row_count == 0:
                    raise ValueError(f"should be more than 0 rows for '{table.__tablename__}'")
                print(f"Table '{table.__tablename__}' exists with {row_count} rows.")
            except exc.OperationalError as e:
                print(f"Table '{table.__tablename__}' does not exist. Error: {e}")
            except exc.ProgrammingError as e:
                db.session.rollback()  # Rollback the session to a clean state
                print(f"Table '{table.__tablename__}' does not exist. Error: {e}")

    def before_first_request_instantiate(app):
        from src.data_structures.approximate_nearest_neighbor.two_tower_ann import (
            instantiate_indexes,
        )

        print("INSTANTIATING ALL TEAMS ANNs")
        instantiate_indexes()
        print("INSTANTIATED INDEXES FOR TEAMS")

        print("instantiating user based collabertive filter objects")
        teams = app.config.get("TEAMS_TO_RUN_FOR")
        for team in teams:
            print(f"doing {team}")
            module_path = f"src.data_structures.user_based_recommender.{team}.UserBasedRecommender"
            TeamSpecificUserBasedRecommender = __import__(module_path, fromlist=['UserBasedRecommender']).UserBasedRecommender 
            try:
                TeamSpecificUserBasedRecommender() # initialize singleton 
            except Exception as e:
                print(f"Failed to do user based recommender for {team}, {e}")
                print(traceback.format_exc())
            print(f"done {team}")
        print("instantiated collabertive filter object for teams")

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
        before_first_request_checks()
        before_first_request_instantiate(app)
        print("FULLY DONE INSTANTIATION USE THE APP")
    return app
