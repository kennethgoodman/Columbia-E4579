# models.py

import datetime
import os

import jwt
from flask import current_app
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src import bcrypt, db
from src.api.content.models import Content

# As per https://stackoverflow.com/questions/63542818/mypy-and-inheriting-from-a-class-that-is-an-attribute-on-an-instance
BaseModel: DeclarativeMeta = db.Model


class User(BaseModel):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(128), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean(), default=True, nullable=False)
    created_date = db.Column(db.DateTime, default=func.now(), nullable=False)

    # relationship
    authored_content = relationship("Content", back_populates="author", uselist=True)

    def __init__(self, username="", password=""):
        self.username = username
        self.password = bcrypt.generate_password_hash(
            password, current_app.config.get("BCRYPT_LOG_ROUNDS")
        ).decode()

    def encode_token(self, user_id, token_type):
        if token_type == "access":
            seconds = current_app.config.get("ACCESS_TOKEN_EXPIRATION")
        else:
            seconds = current_app.config.get("REFRESH_TOKEN_EXPIRATION")

        payload = {
            "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=seconds),
            "iat": datetime.datetime.utcnow(),
            "sub": user_id,
        }
        return jwt.encode(
            payload, current_app.config.get("SECRET_KEY"), algorithm="HS256"
        )

    @staticmethod
    def decode_token(token):
        payload = jwt.decode(
            token, current_app.config.get("SECRET_KEY"), algorithms="HS256"
        )
        return payload["sub"]


if os.getenv("FLASK_ENV") == "development":
    from src import admin
    from src.api.users.admin import UsersAdminView

    admin.add_view(UsersAdminView(User, db.session))
