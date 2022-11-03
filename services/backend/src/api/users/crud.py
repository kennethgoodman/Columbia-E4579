from src import db
from src.api.users.models import User


def get_all_users():
    return User.query.all()


def get_user_by_id(user_id):
    return User.query.filter_by(id=user_id).first()


def get_user_by_username(username):
    return User.query.filter_by(username=username).first()


def add_user(username, password):
    user = User(username=username, password=password)
    with db.session() as session:
        session.add(user)
        session.commit()
    return user


def update_user(user, username):
    user.username = username
    with db.session() as session:
        session.commit()
    return user


def delete_user(user):
    with db.session() as session:
        session.delete(user)
        session.commit()
    return user
