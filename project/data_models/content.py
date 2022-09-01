# content.py

from project import db
from sqlalchemy import Enum as SqlEnum, ForeignKey
from sqlalchemy.orm import relationship


from enum import Enum


def get_url(content):
    return f"https://{content.s3_bucket}.s3.amazonaws.com/{content.s3_id}"


class MediaType(Enum):
    Image = 1
    Text = 2
    Video = 3


class Content(db.Model):
    __tablename__ = "content"
    id = db.Column(db.Integer,
                   ForeignKey("generated_content_metadata.id"),
                   ForeignKey("non_generated_content_metadata.id"),
                   primary_key=True
    )  # primary keys are required by SQLAlchemy
    media_type = db.Column(SqlEnum(MediaType))
    s3_bucket = db.Column(db.String(200), nullable=True)
    s3_id = db.Column(db.String(200), nullable=True)  # might be only text, if media_type = Text

    # Foreign Keys
    author_id = db.Column(db.Integer, ForeignKey("user.id"), nullable=False)
    # all engagements on the content
    content_engagements = relationship("Engagement")  # one piece of content with many engagements
