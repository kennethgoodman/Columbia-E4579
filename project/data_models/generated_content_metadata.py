# generated_content_metadata.py

from project import db
from sqlalchemy import ForeignKey


class GeneratedContentMetadata(db.Model):
    __tablename__ = "generated_content_metadata"
    id = db.Column(db.Integer, ForeignKey("content.id"), primary_key=True)  # primary keys are required by SQLAlchemy
    seed = db.Column(db.Integer)
    num_inference_steps = db.Column(db.Integer)
    guidance_scale = db.Column(db.Integer)
    prompt = db.Column(db.String(1500))
    original_prompt = db.Column(db.String(1500))
    artist_style = db.Column(db.String(100))
    source = db.Column(db.String(100))
