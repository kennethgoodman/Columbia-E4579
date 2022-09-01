# non_generated_content_metadata.py

from project import db
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship


class NonGeneratedContentMetadata(db.Model):
    __tablename__ = "non_generated_content_metadata"
    id = db.Column(db.Integer, primary_key=True)  # primary keys are required by SQLAlchemy
    content_id = db.Column(db.Integer, ForeignKey("content.id"))
    content = relationship("Content", back_populates="generated_content_metadata")

    source = db.Column(db.String(100))
    text = db.Column(db.String(1000), nullable=True)  # text on the post
