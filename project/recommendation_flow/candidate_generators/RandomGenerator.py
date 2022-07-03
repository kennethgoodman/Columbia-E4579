from . import AbstractGenerator
from project.models.content import Content
from  sqlalchemy.sql.expression import func


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self):
        results = Content.query.limit(1000).order_by(func.rand()).all()
        return list(map(lambda x: x.id, results))
