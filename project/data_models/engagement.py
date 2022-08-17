# engagement.py

from project import db
from sqlalchemy import Enum as SqlEnum, ForeignKey
from enum import Enum


class EngagementType(Enum):
    Like = 1
    Comment = 2
    VideoWatch = 3  # more than 100ms


class Engagement(db.Model):
    __tablename__ = "engagement"
    id = db.Column(db.Integer, primary_key=True)  # primary keys are required by SQLAlchemy
    user_id = db.Column(db.Integer, ForeignKey('user.id'))  # user that engaged
    content_id = db.Column(db.Integer, ForeignKey('content.id'))  # content they engaged with
    engagement_type = db.Column(SqlEnum(EngagementType))  # how they engaged
