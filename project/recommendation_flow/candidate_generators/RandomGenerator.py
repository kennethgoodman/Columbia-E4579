from .AbstractGenerator import AbstractGenerator
from project.models.content import Content
from sqlalchemy.sql.expression import func


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self):
        results = Content.query.with_entities(Content.id).order_by(func.rand()).limit(100).all()
        return list(map(lambda x: x[0], results))
