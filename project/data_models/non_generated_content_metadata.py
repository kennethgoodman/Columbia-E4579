# non_generated_content_metadata.py

from project import db
from sqlalchemy import ForeignKey


class NonGeneratedContentMetadata(db.Model):
    __tablename__ = "non_generated_content_metadata"
    id = db.Column(db.Integer, ForeignKey("content.id"), primary_key=True)  # primary keys are required by SQLAlchemy
    source = db.Column(db.String(100))
