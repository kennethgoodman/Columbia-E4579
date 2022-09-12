import csv
import os

import boto3
from botocore.exceptions import ClientError
from project.data_models.content import Content, MediaType
from project.data_models.generated_content_metadata import (
    GeneratedContentMetadata,
    GeneratedType,
)

from project import create_app, db
from project.data_models import User, _tables

app = create_app()

s3 = boto3.resource("s3")
s3_client = boto3.client("s3")
buckets = list(s3.buckets.all())


def publish_content_for_user(s3_bucket, key, author_id, row, generation_params):
    # create new user with the form data. Hash the password so plaintext version isn't saved.
    new_content = Content(
        media_type=MediaType.Image, s3_bucket=s3_bucket, s3_id=key, author_id=author_id
    )

    new_metadata = GeneratedContentMetadata(
        content=new_content,
        seed=generation_params["seed"],
        num_inference_steps=generation_params["num_inference_steps"],
        guidance_scale=generation_params["guidance_scale"],
        prompt=row["prompt"],
        original_prompt=row["original_prompt"],
        artist_style=row["artist_style"],
        source=row["source"],
        source_img=row.get("source_img", None),
        generated_type=GeneratedType.HumanTxt2Img,
    )

    # add the new user to the database
    db.session.add(new_content)
    db.session.add(new_metadata)
    db.session.commit()
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
