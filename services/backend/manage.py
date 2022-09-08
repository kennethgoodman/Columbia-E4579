from flask.cli import FlaskGroup

from src import create_app, db
from src.api.users.models import User
from src.api.content.models import Content, MediaType


app = create_app()
cli = FlaskGroup(create_app=create_app)


@cli.command("recreate_db")
def recreate_db():
    db.drop_all()
    db.create_all()
    db.session.commit()


@cli.command('seed_db')
def seed_db():
    tmp_1 = User(username='abc', password="supersecret")
    db.session.add(tmp_1)
    tmp_2 = User(username='xyz', password="supersecret")
    db.session.add(tmp_2)
    db.session.add(Content(media_type=MediaType.Image, id=0, author=tmp_2))
    db.session.commit()


if __name__ == "__main__":
    cli()
