import csv
import os
import pickle
import random

from flask.cli import FlaskGroup
from src import create_app, db
from src.api.content.models import (
    Content,
    GeneratedContentMetadata,
    MediaType,
    ModelType,
)
from src.api.users.models import User

cli = FlaskGroup(create_app=create_app)


@cli.command("recreate_db")
def recreate_db():
    db.drop_all()
    db.create_all()
    db.session.commit()


@cli.command("seed_db")
def seed_db():
    users = []
    print("seeding users")
    with open("seed_data/data/users.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            user = User(**row)
            users.append(user)
            db.session.add(user)
    print("reading embeddings")
    with open("seed_data/data/prompt_to_embedding.64.100.1000.pkl", "rb") as f:
        prompt_to_embedding = pickle.load(f)
    print("seeded content with metadata")
    with open("seed_data/data/content_with_metadata.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=chr(255))
        for row in reader:
            content = Content(
                media_type=MediaType.Image,
                author=random.choice(users),
                s3_bucket=row["s3_bucket"],
                s3_id=row["s3_id"],
            )
            db.session.add(content)
            metadata = GeneratedContentMetadata(
                content=content,
                original_prompt=row["original_prompt"],
                source=row["source"],
                artist_style=row["artist_style"],
                seed=row["seed"],
                num_inference_steps=row["num_inference_steps"],
                guidance_scale=row["guidance_scale"],
                prompt=row["prompt"],
                source_img=row["source_img"],
                generated_type=row["generated_type"],
                model=ModelType.StableDiffusion,  # TODO: read this, don't hardcode
                model_version="1.4",  # TODO: read this, don't hardcode
                prompt_embedding=list(prompt_to_embedding.get(row["prompt"], []))
                or None,
            )
            db.session.add(metadata)
    print("committing data")
    db.session.commit()  # TODO: explore committing more often


if __name__ == "__main__":
    cli()
