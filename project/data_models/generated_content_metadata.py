# generated_content_metadata.py

from project import db
from sqlalchemy import Enum as SqlEnum, ForeignKey
from sqlalchemy.orm import relationship
from enum import Enum


class GeneratedType(Enum):
    HumanTxt2Img = 1
    GPT3Txt2Img = 2
    Img2Txt2Img = 3
    Img2Img = 4


class GeneratedContentMetadata(db.Model):
    __tablename__ = "generated_content_metadata"
    id = db.Column(db.Integer, primary_key=True)  # primary keys are required by SQLAlchemy
    content_id = db.Column(db.Integer, ForeignKey("content.id"))
    content = relationship("Content", back_populates="generated_content_metadata")

    seed = db.Column(db.Integer)
    num_inference_steps = db.Column(db.Integer)
    guidance_scale = db.Column(db.Integer)
    prompt = db.Column(db.String(1500))
    original_prompt = db.Column(db.String(1500))
    artist_style = db.Column(db.String(100))
    source = db.Column(db.String(100))
    source_img = db.Column(db.String(200), nullable=True)
    generated_type = db.Column(SqlEnum(GeneratedType))
