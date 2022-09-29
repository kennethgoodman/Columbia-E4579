import operator

from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator


class EngagementTimeGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if starting_point is None:
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
            num_results = len(results)
            if num_results == 0:
                return RandomGenerator().get_content_ids(
                    user_id, limit, offset, seed, starting_point
                )
            new_limit = 2 * (limit // num_results + 1)  # get 2x so we can take the best
            new_offset = offset // num_results + 1
            new_result, new_scores = [], []
            for (content_id, ms) in results:
                # can probably do this faster as a heap so instead of `2 * limit * (1+log2(limit))` in `limit * log2(limit)`
                content_ids, scores = ann_with_offset(
                    content_id, 0.9, 2 * new_limit, new_offset, return_distances=True
                )
                new_result.extend(content_ids)
                # TODO: score and ms should probably be normalized so multiplication makes sense
                new_scores.extend(map(lambda score: score * ms, scores))
            if len(new_result) == 0:
                return RandomGenerator().get_content_ids(
                    user_id, limit, offset, seed, starting_point
                )
            results_with_scores = sorted(
                list(zip(new_result, new_scores)), key=operator.itemgetter(1)
            )[:limit]
            return list(map(operator.itemgetter(0), results_with_scores)), list(
                map(operator.itemgetter(1), results_with_scores)
            )
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
