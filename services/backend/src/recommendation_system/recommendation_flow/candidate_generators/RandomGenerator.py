from .AbstractGenerator import AbstractGenerator
from src.api.content.models import Content
from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self, limit, offset, seed):
        sql = text('select setseed({0});'.format(seed))
        db.engine.execute(sql)
        results = Content.query.with_entities(Content.id).order_by(func.random()).limit(limit).offset(offset).all()
        return list(map(lambda x: x[0], results))
