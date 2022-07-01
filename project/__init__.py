# init.py
import logging
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from project.logging_utils.config import configure_logging

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()


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
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from project.models.user import User

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