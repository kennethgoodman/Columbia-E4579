from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator


class RandomGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if starting_point is None:
            results = (
                Content.query.with_entities(Content.id)
                .order_by(func.random(seed))
                .limit(limit)
                .offset(offset)
                .all()
            )
            return list(map(lambda x: x[0], results)), None
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
