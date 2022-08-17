# engagement.py

from project import db
from sqlalchemy import Enum as SqlEnum, ForeignKey
from enum import Enum


class EngagementType(Enum):
    Like = 1
    Comment = 2
    VideoWatch = 3  # more than 100ms


class Engagement(db.Model):
    __table_args__ = (
        # this can be db.PrimaryKeyConstraint if you want it to be a primary key
        db.UniqueConstraint('user_id', 'content_id', 'engagement_type'),
    )
    __tablename__ = "engagement"
    id = db.Column(db.Integer, primary_key=True)  # primary keys are required by SQLAlchemy
    user_id = db.Column(db.Integer, ForeignKey('user.id'))  # user that engaged
    content_id = db.Column(db.Integer, ForeignKey('content.id'))  # content they engaged with
    engagement_type = db.Column(SqlEnum(EngagementType))  # how they engaged
