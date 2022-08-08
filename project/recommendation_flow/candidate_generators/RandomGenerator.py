from .AbstractGenerator import AbstractGenerator
from project.data_models.content import Content
from sqlalchemy.sql.expression import func


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self, limit):
        results = Content.query.with_entities(Content.id).order_by(func.rand()).limit(limit).all()
        return list(map(lambda x: x[0], results))
