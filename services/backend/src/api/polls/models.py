from src import db
from src.api.utils.auth_utils import get_user
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import relationship

# As per https://stackoverflow.com/questions/63542818/mypy-and-inheriting-from-a-class-that-is-an-attribute-on-an-instance
BaseModel: DeclarativeMeta = db.Model

class Poll(BaseModel):
    __tablename__ = "polls"
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500), nullable=False)
    available = db.Column(db.Boolean, default=False, nullable=False)
    choices = relationship("Choice", back_populates='poll', cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates='poll', cascade="all, delete-orphan")


class Choice(BaseModel):
    __tablename__ = "choices"
    id = db.Column(db.Integer, primary_key=True)
    poll_id = db.Column(db.Integer, db.ForeignKey("polls.id"), nullable=False)
    text = db.Column(db.String(200), nullable=False)
    poll = relationship("Poll", back_populates="choices")
    votes = relationship("Vote", back_populates='choice', cascade="all, delete-orphan")
    

class Vote(BaseModel):
    __tablename__ = "votes"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    poll_id = db.Column(db.Integer, db.ForeignKey('polls.id'), nullable=False)
    choice_id = db.Column(db.Integer, db.ForeignKey('choices.id'), nullable=False)
    
    # Define relationships if needed
    user = relationship("User", backref="votes")
    poll = relationship("Poll", back_populates="votes")
    choice = relationship("Choice", back_populates="votes")
