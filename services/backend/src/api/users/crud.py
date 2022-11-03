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
    db.session.add(user)
    db.session.commit()
    return user


def update_user(user, username):
    user.username = username
    db.session.commit()
    return user


def delete_user(user):
    db.session.delete(user)
    db.session.commit()
    return user
