# user.py

from flask_login import UserMixin
from project import db
from sqlalchemy.orm import relationship


class User(UserMixin, db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)  # primary keys are required by SQLAlchemy
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    posted_content = relationship("Content")  # 1 user to many pieces of content
    engagements = relationship("Engagement")
