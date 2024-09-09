from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content, MediaType

from .AbstractGenerator import AbstractGenerator


class RandomGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        results = (
            Content.query.with_entities(Content.id)
            .filter(func.mod(func.rand(seed + Content.id), 10) < 1)
            .order_by(func.random(seed))
            .limit(limit)
            .offset(offset)
            .all()
        )
        return list(map(lambda x: x[0], results)), [1.0] * len(results)
    
    def _get_name(self):
        return "Random"

class RandomGeneratorText(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        results = (
            Content.query.with_entities(Content.id)
            .filter_by(
                media_type=MediaType.Text,
            )
            .filter(func.mod(func.rand(seed + Content.id), 10) < 1)
            .order_by(func.random(seed))
            .limit(limit)
            .offset(offset)
            .all()
        )
        return list(map(lambda x: x[0], results)), [1.0] * len(results)
    
    def _get_name(self):
        return "RandomText"


class RandomGeneratorImage(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        results = (
            Content.query.with_entities(Content.id)
            .filter_by(
                media_type=MediaType.Image,
            )
            .filter(func.mod(func.rand(seed + Content.id), 10) < 1)
            .order_by(func.random(seed))
            .limit(limit)
            .offset(offset)
            .all()
        )
        return list(map(lambda x: x[0], results)), [1.0] * len(results)
    
    def _get_name(self):
        return "RandomImage"
