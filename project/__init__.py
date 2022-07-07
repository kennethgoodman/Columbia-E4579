# init.py
import logging
from os import environ
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from project.logging_utils.config import configure_logging
from dotenv import dotenv_values

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()


def get_sqlalchemy_database_uri():
    config = dotenv_values(".env")
    if int(config.get('use_aws_db', 0)) == 1:
        # all must exist or will throw exception
        username, password = config['aws_db_username'], config['aws_db_password']
        databaseurl, schema = config['aws_db_endpoint'], config['aws_db_schema']
        return f'mysql://{username}:{password}@{databaseurl}/{schema}'
    # if don't use AWS, then use SQLite
    return 'sqlite:///db.sqlite'


# we set up proxy so we can do front end dev if we need and serve to react local server
def proxy(path):
    from requests import get
    print("in proxy")
    host = "http://localhost:3000"
    response = get(f"{host}{path}")
    print(response.content)
    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    headers = {
        name: value
        for name, value in response.raw.headers.items()
        if name.lower() not in excluded_headers
    }
    return response.content, response.status_code, headers


def create_app():
    if not os.path.isdir('logs_folder'):
        os.mkdir('logs_folder')
    error_logger = configure_logging('logs_folder/error.log', logging.WARN, 'error_log')
    info_logger = configure_logging('logs_folder/info.log', logging.INFO, 'info_log')
    debug_logger = configure_logging('logs_folder/debug.log', logging.DEBUG, 'debug_log')
    loggers = [error_logger, info_logger, debug_logger]
    # to load the flask data, set static folder to build
    app = Flask(__name__, static_folder='project/frontend/build')

    # add them all together
    app.logger.handlers = list(sum(map(lambda x: x.handlers, loggers), []))
    # take the minimum level
    app.logger.setLevel(min(map(lambda x: x.level, loggers)))

    app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
    app.config['FLASK_DEBUG'] = int(environ.get("FLASK_DEBUG", 0))
    app.config['SQLALCHEMY_DATABASE_URI'] = get_sqlalchemy_database_uri()
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.proxy = proxy

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from project.data_models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from project.backend_routes.auth import auth as auth_blueprint
    from project.backend_routes.main import main as main_blueprint
    from project.backend_routes.data_api import data_api
    blueprints = [auth_blueprint, main_blueprint, data_api]
    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run()
