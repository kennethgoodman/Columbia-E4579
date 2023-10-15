from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import (
    AbstractGenerator,
)
from sqlalchemy.sql.expression import func
from sqlalchemy.sql import text, select

from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
import pandas as pd


class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if starting_point is None:
            # nb of likes subquery
            results_like = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter(Engagement.engagement_type == EngagementType.Like)
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                .limit(int(limit * 5))
                .offset(offset)
                .all()
            )

            # engagement time subquery
            results_engage_time = (
                Engagement.query.with_entities(
                    Engagement.content_id, Engagement.engagement_value
                )
                .filter(Engagement.engagement_value <= 20000)
                .filter(Engagement.engagement_value >= 1000)
                .order_by(Engagement.engagement_value.desc())
                .limit(
                    int(limit * 5)
                )  # * 5  to take into account filtering on both list will eventually decrease the nb of candidates by quite a lot
                .offset(offset)
                .all()
            )

            # Get content_ids from results_engage_time
            print("results like", len(results_like))
            print("results engage", len(results_engage_time))
            engage_time_content_ids = [result[0] for result in results_engage_time]

            # Filter results_like based on these content_ids
            results_like_new = [
                result
                for result in results_like
                if result[0] in engage_time_content_ids
            ]
            print("Num of candidates", len(results_like_new))
            # Create a dictionary where the keys are the content_ids and the values are tuples of (number_of_likes, engagement_time)
            engagement_data = {result[0]: (result[1], 0) for result in results_like_new}
            for result in results_engage_time:
                if result[0] in engagement_data:
                    engagement_data[result[0]] = (
                        engagement_data[result[0]][0],
                        result[1],
                    )

            # Compute the engagement score for each content_id
            results_engagement_score = [
                (content_id, likes * time)
                for content_id, (likes, time) in engagement_data.items()
            ]

            results = list(set(results_engagement_score))
            print("Num of Candidates:", len(results))

            return list(map(lambda x: x[0], results)), list(
                map(lambda x: x[1], results)  # engagement_score
            )

        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")

        # # NEXT POTENTIAL IDEA - build K Means clusters and pick best candidate based on User AND Content

    def _get_name(self):
        return "YourChoiceGenerator"
