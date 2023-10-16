from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
from src.api.engagement.models import Engagement, EngagementType


from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator

class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if starting_point is None:
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
            return list(map(lambda x: x[0], results)), [0]*(len(results))
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
    def _get_name(self):
        return "YourChoiceGenerator"
