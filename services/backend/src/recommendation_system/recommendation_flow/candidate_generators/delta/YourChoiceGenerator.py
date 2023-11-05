import operator
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator


class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        if starting_point.get("content_id", None) is None:
            # Create a list to select the content based on the number of "like" they gain from previous users.
            results_popular = (
                Engagement.query.with_entities(
                    Engagement.content_id, func.count()
                )
                .filter_by(
                    engagement_type=EngagementType.Like,
                    engagement_value=1
                )
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
            Example_r = list(map(lambda x: x[0], results_popular))

            # Select the items in "MillisecondsEngagedWith" to choose some items not rated by the user yet.
            # For the content we select, we order them by the Engagement_Value since longer time of engagement may mean a higher probability users will like.
            results_top = (
                Engagement.query.with_entities(
                    Engagement.content_id, func.sum(Engagement.engagement_value)
                )
                .filter_by(
                    engagement_type=EngagementType.MillisecondsEngagedWith,
                )
                .group_by(Engagement.content_id)
                .order_by(func.sum(Engagement.engagement_value).desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            # Extract content IDs from results_top
            Example_t = list(map(lambda x: x[0], results_top))

            # Combine the content we get from two selections.
            # In order to relieve consumers' visual fatigue, we make this new list have a picture with a higher engagement value appear every five popular pictures to give users a novel feeling.
            results = []
            for i, item in enumerate(Example_r):
                results.append(item)
                if i % 5 == 4 and i < len(Example_r) - 1:
                    results.append(Example_t[i // 5])
            scores = [i for i in range(len(results),-1,-1)]  

            return results, scores
        content_ids, scores = ann_with_offset(
            starting_point["content_id"], 0.9, limit, offset, return_distances=True
        )
        return content_ids, scores

    def _get_name(self):
        return "YourChoiceGenerator"

