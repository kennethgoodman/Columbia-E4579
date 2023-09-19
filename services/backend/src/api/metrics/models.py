# models.py

from enum import Enum, IntEnum, unique

from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src import db

# As per https://stackoverflow.com/questions/63542818/mypy-and-inheriting-from-a-class-that-is-an-attribute-on-an-instance
BaseModel: DeclarativeMeta = db.Model

@unique
class MetricFunnelType(Enum):
    CandidateGeneration = 1
    Filtering = 2
    Prediction = 3
    Ranking = 4
    Controller = 5

@unique
class MetricType(Enum):
    CandidateGenerationNumCandidates = 1

@unique
class TeamName(Enum):
    Example = 1
    Random = 2
    EngagementTime = 3
    Alpha_F2023 = 4
    Beta_F2023 = 5
    Charlie_F2023 = 6
    Delta_F2023 = 7
    Echo_F2023 = 8
    Foxtrot_F2023 = 9
    Golf_F2023 = 10


class Metric(BaseModel):
    __tablename__ = "metric"
    id = db.Column(
        db.Integer, primary_key=True
    )  # primary keys are required by SQLAlchemy
    team_name = db.Column(SqlEnum(TeamName), nullable=False)
    funnel_name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, ForeignKey("user.id"), nullable=True)
    content_id = db.Column(
        db.Integer, ForeignKey("content.id"), nullable=True
    )  # content they engaged with
    metric_funnel_type = db.Column(SqlEnum(MetricFunnelType), nullable=False)
    metric_type = db.Column(SqlEnum(MetricType), nullable=False)
    metric_value = db.Column(
        db.Integer, nullable=True
    )  # the value of the engagement_type
    created_date = db.Column(db.DateTime, default=func.now(), nullable=False)
    metric_metadata = db.Column(db.JSON, nullable=True)


