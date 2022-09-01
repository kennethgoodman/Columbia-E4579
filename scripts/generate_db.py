from project import db, create_app
from project.data_models import _tables
import sqlalchemy.exc
from dotenv import dotenv_values


def check_if_production():
    config = dotenv_values(".env")
    if int(config.get('use_aws_db', 0)) == 1:
        print("YOU ARE ABOUT TO DELETE PRODUCTION, ARE YOU SURE?")
        user_input = input("(yes/no): ")
        if user_input != "yes":
            print("you didn't write 'yes', so quitting")
            exit()


def delete_old_db():
    check_if_production()
    for table in _tables:
        try:
            table.__table__.drop(db.engine)
        except sqlalchemy.exc.OperationalError:
            pass
    db.session.commit()


def main():
    app = create_app()
    with app.app_context():
        delete_old_db()
    db.create_all(app=app)


if __name__ == '__main__':
    main()
