import csv
import os
import random

from flask.cli import FlaskGroup
from src import create_app, db
from src.api.users.models import User
from src.api.content.models import Content, MediaType, GeneratedContentMetadata, ModelType

app = create_app()
cli = FlaskGroup(create_app=create_app)


@cli.command("recreate_db")
def recreate_db():
    db.drop_all()
    db.create_all()
    db.session.commit()


@cli.command("seed_db")
def seed_db():
    users = []
    with open("seed_data/data/users.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            user = User(**row)
            users.append(user)
            db.session.add(user)
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
                original_prompt=row['original_prompt'],
                source=row['source'],
                artist_style=row['artist_style'],
                seed=row['seed'],
                num_inference_steps=row['num_inference_steps'],
                guidance_scale=row['guidance_scale'],
                prompt=row['prompt'],
                source_img=row['source_img'],
                generated_type=row['generated_type'],
                model=ModelType.StableDiffusion, # TODO: read this, don't hardcode
                model_version='1.4', # TODO: read this, don't hardcode
            )
            db.session.add(metadata)
    db.session.commit()


@cli.command('recreate_and_seed_db')
def recreate_and_seed_db():
    recreate_db()
    seed_db

if __name__ == "__main__":
    cli()
