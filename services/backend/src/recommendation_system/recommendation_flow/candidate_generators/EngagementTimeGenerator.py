import operator

from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator


class EngagementTimeGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        # TODO: should discount by creation_time so closer events have more weight
        results = (
            Engagement.query.with_entities(
                Engagement.content_id, func.max(Engagement.engagement_value)
            )
            .filter_by(
                user_id=user_id,
                engagement_type=EngagementType.MillisecondsEngagedWith,
            )
            .group_by(Engagement.content_id)
            .order_by(func.max(Engagement.engagement_value))
            .limit(limit + 1)
            .offset(offset)
            .all()
        )
        if len(results) == 0:
            return RandomGenerator()._get_content_ids(
                user_id, limit, offset, seed, starting_point
            )
        return results, [1.0] * num_results

    def _get_name(self):
        return "EngagementTime"
