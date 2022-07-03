# content.py

from project import db
from sqlalchemy import Enum as SqlEnum, ForeignKey
from sqlalchemy.orm import relationship


from enum import Enum


class MediaType(Enum):
    Image = 1
    Text = 2
    Video = 3


class Content(db.Model):
    __tablename__ = "content"
    id = db.Column(db.Integer, primary_key=True)  # primary keys are required by SQLAlchemy
    media_type = db.Column(SqlEnum(MediaType))
    s3_id = db.Column(db.String(100), nullable=True)  # might be only text, if media_type = Text
    text = db.Column(db.String(1000))  # text on the post
    author_id = db.Column(db.Integer, ForeignKey("user.id"))
    # all engagements on the content
    content_engagements = relationship("Engagement")  # one piece of content with many engagements
