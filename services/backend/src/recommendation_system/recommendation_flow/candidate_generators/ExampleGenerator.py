import operator
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator


class ExampleGenerator(AbstractGenerator):
    def get_content_ids(self, _, limit, offset, _seed, starting_point):
        if starting_point is None:
            # TODO: should discount by creation_time so closer events have more weight
            results = (
                Engagement.query.with_entities(
                    Engagement.content_id, func.count()
                )
                .filter_by(
                    engagement_type=EngagementType.Like,
                )
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
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
