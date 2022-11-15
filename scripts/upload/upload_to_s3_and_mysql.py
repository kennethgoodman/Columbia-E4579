import csv
import os

import boto3
from botocore.exceptions import ClientError
from sqlalchemy.orm import sessionmaker
from src import create_app, db
from src.api.content.models import (
    Content,
    GeneratedContentMetadata,
    GeneratedType,
    MediaType,
    ModelType,
)
from src.api.users.models import User

app = create_app()

s3 = boto3.resource("s3")
s3_client = boto3.client("s3")
buckets = list(s3.buckets.all())
import csv
import pickle
import time


def publish_content_for_user(author_id, prompt_to_embedding, **kwargs):
    # create new user with the form data. Hash the password so plaintext version isn't saved.
    new_content = Content(
        media_type=MediaType.Image,
        s3_bucket=kwargs["s3_bucket"],
        s3_id=kwargs["key"],
        author_id=author_id,
    )
    new_metadata = GeneratedContentMetadata(
        content=new_content,
        seed=int(kwargs["seed"]),
        num_inference_steps=int(kwargs["num_inference_steps"]),
        guidance_scale=float(kwargs["guidance_scale"]),
        prompt=kwargs["prompt"],
        original_prompt=kwargs["original_prompt"],
        artist_style=kwargs["artist_style"],
        source=kwargs["source"],
        source_img=kwargs["source_img"],
        generated_type=GeneratedType(int(kwargs["generation_type"])),
        model=ModelType.StableDiffusion,
        model_version="1.4",
        prompt_embedding=list(prompt_to_embedding.get(kwargs["prompt"], [])) or None,
    )
    # add the new user to the database
    with db.session() as session:
        session.add(new_content)
        session.add(new_metadata)
        session.commit()
    print("committed")


def get_author_id(username):
    return User.query.filter_by(username=username).first()


def upload_to_s3(file_name, bucket, object_name):
    try:
        bucket.upload_file(file_name, object_name)
    except ClientError as e:
        print(e)
        return None
    return file_name


def get_dir_from_generation_params(generation_params):
    return os.path.join(
        "/Users/kennethgoodman/Downloads/reddit/"
        f'{generation_params["num_inference_steps"]}',
        f'{generation_params["seed"]}',
        f'{generation_params["guidance_scale"]}',
    )


def get_filename_from_generation_params(idx, generation_params):
    fn = f"{idx}.png"
    return os.path.join(get_dir_from_generation_params(generation_params), fn), fn


def get_object_name_from_generation_params(idx, generation_params):
    return f'{generation_params["num_inference_steps"]}_{generation_params["seed"]}_{generation_params["guidance_scale"]}_{idx}.png'


def try_publish(author_id, prompt_to_embedding, info):
    try:
        publish_content_for_user(author_id, prompt_to_embedding, **info)
    except Exception as e:
        if "Duplicate entry" in str(e):
            print("duplicate entry, moving on", str(e), str(info))
            return
        raise e  # don't know whats wrong


def write_to_database(author_id, start_from=0, end_at=None):
    with open("/home/ec2-user/prompt_to_embedding.512.100.1000.pkl", "rb") as f:
        prompt_to_embedding = pickle.load(f)
    with open("/home/ec2-user/columbia_e4579_images.csv") as csvfile:
        rows = list(csv.DictReader(csvfile))[start_from:]
        if end_at:
            rows = rows[: end_at - start_from]
    with app.app_context():
        for i, info in enumerate(rows):
            print("doing", i + start_from)
            try_publish(author_id, prompt_to_embedding, info)


def main(generation_params):
    bucket = list(s3.buckets.all())[0]
    files = os.listdir(get_dir_from_generation_params(generation_params))
    with app.app_context():
        author_id = get_author_id("kgoodman").id
        with open(
            "image_generation_pipelines/output/reddit.csv", newline="", encoding="utf-8"
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                print("doing", i)
                full_path, fn = get_filename_from_generation_params(
                    i, generation_params
                )
                if fn not in files:
                    print("not doing", i, "because it doesn't exist")
                    continue
                object_name = get_object_name_from_generation_params(
                    i, generation_params
                )
                s3_id = upload_to_s3(
                    full_path, bucket, f"generated_images/{object_name}"
                )
                if s3_id is None:
                    print("failed", row)
                    continue
                publish_content_for_user(
                    bucket.name,
                    f"generated_images/{object_name}",
                    author_id,
                    row,
                    generation_params,
                )


if __name__ == "__main__":
    main(
        generation_params={
            "seed": 7,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        }
    )
