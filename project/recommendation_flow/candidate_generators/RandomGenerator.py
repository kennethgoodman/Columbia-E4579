from .AbstractGenerator import AbstractGenerator
from project.data_models.content import Content
from sqlalchemy.sql.expression import func
from flask import session


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self, limit, offset):
        seed = int(session.get('session_id'))
        results = Content.query.with_entities(Content.id).order_by(func.rand(seed)).limit(limit).offset(offset).all()
        return list(map(lambda x: x[0], results))
