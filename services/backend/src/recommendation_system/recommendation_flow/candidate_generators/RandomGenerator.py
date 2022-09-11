from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content

from .AbstractGenerator import AbstractGenerator


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self, limit, offset, seed):
        sql = text(f"select setseed({seed});")
        db.engine.execute(sql)
        results = (
            Content.query.with_entities(Content.id)
            .order_by(func.random())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return list(map(lambda x: x[0], results))
