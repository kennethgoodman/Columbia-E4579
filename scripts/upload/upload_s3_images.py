import boto3
from project import create_app, db
from project.data_models import User, _tables
from project.data_models.content import Content, MediaType

app = create_app()

s3 = boto3.resource("s3")
buckets = list(s3.buckets.all())


def publish_content_for_user(s3_bucket, key, text, author_id):
    # create new user with the form data. Hash the password so plaintext version isn't saved.
    new_content = Content(
        media_type=MediaType.Image, s3_bucket=s3_bucket, s3_id=key, author_id=author_id
    )

    # add the new user to the database
    db.session.add(new_content)


def get_author_id(username):
    return User.query.filter_by(username=username).first()


def main():
    with app.app_context():
        author_id = get_author_id("kgoodman").id
        for bucket in buckets:
            for i, obj in enumerate(bucket.objects.all()):
                key = obj.key
                bucket_name = bucket.name
                publish_content_for_user(bucket_name, key, "", author_id)

                if i % 100 == 1:
                    print(i, "finished writing sql, not at committing")
                    db.session.commit()  # all or nothing
                    print("committed")
        print("finished writing sql, not at committing")
        db.session.commit()  # all or nothing
        print("committed")


if __name__ == "__main__":
    main()
