# models.py

from enum import Enum, IntEnum, unique

from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src import db


@unique
class LikeDislike(IntEnum):
    Like = 1
    Dislike = -1


@unique
class EngagementType(Enum):
    Like = 1
    Comment = 2
    VideoWatch = 3  # more than 100ms
    ImageView = 4  # more than 75% of the photo for > 200ms
    MillisecondsEngagedWith = 5


class Engagement(db.Model):
    __tablename__ = "engagement"
    id = db.Column(
        db.Integer, primary_key=True
    )  # primary keys are required by SQLAlchemy
    user_id = db.Column(db.Integer, ForeignKey("user.id"))  # user that engaged
    content_id = db.Column(
        db.Integer, ForeignKey("content.id")
    )  # content they engaged with
    engagement_type = db.Column(SqlEnum(EngagementType))  # how they engaged
    engagement_value = db.Column(
        db.Integer, nullable=True
    )  # the value of the engagement_type
    created_date = db.Column(db.DateTime, default=func.now(), nullable=False)
