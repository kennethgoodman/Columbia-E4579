import operator
import pandas as pd
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator

df_user_clusters = pd.read_csv(r"/usr/src/app/src/foxtrot/foxtrot_users_clusters2.csv", nrows=100)

class FoxtrotGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        if starting_point is None:
            # 40% of the content will be found according to engagement metrics
            # 30% through (approximate) nearest neighbours search of liked content
            # 30% from content liked by similar users

            # TODO: should discount by creation_time so closer events have more weight

            # content found according to engagement metrics
            results_engagement = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter_by(
                    engagement_type=EngagementType.Like,
                )
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            list_engagement = list(map(lambda x: x[0], results_engagement))
            # get content most engaged with by user
            results_ANN = []
            # for likes

            table_ann_likes = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter_by(
                    user_id=user_id,
                    engagement_type=EngagementType.Like,
                    engagement_value=1,
                )
                # .where(Engagement.engagement_value==1)
                # .order_by(Engagement.engagement_value.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            # for engagement time

            table_ann_engagement_time = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter_by(
                    user_id=user_id,
                    engagement_type=EngagementType.MillisecondsEngagedWith,
                )
                .order_by(Engagement.engagement_value.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            liked_by_user = list(map(lambda x: x[0], table_ann_likes)) + list(
                map(lambda x: x[0], table_ann_engagement_time)
            )

            # get ANN for the items found
            for liked_content_id in liked_by_user:
                if liked_content_id is None:
                    continue
                content_ids, scores = ann_with_offset(
                    liked_content_id,
                    0.9,
                    limit,
                    offset,
                    return_distances=True,
                )
                results_ANN.append(content_ids)

            # Collaborative filtering: use users clusters to find similar users from user_id, then find most liked items
            # for similar users in cluster.
            results_colaborative = []

            # retrieve users clusters from csv
            users_in_cluster = self.retrieve_cluster(user_id)

            for user in users_in_cluster:
                results, scores = self.get_content_ids_auxilliary(
                    user, limit, offset, _seed, starting_point
                )
                if results is not None:
                    results_colaborative.append(results)

            return list_engagement + results_ANN + results_colaborative, None
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")

    # return list of all users in cluster from csv
    def retrieve_cluster(self, user):
        df = df_user_clusters
        cluster_nbs = df.loc[df["user_id"] == user]["cluster_number"]
        if len(cluster_nbs) == 0:
            return []
        cluster_nb = cluster_nbs.iloc[0]
        users_in_cluster = df.loc[df["cluster_number"] == cluster_nb][
            "user_id"
        ].to_list()
        return users_in_cluster

    # auxilliary function: returns content liked by a user_id in the same cluster
    def get_content_ids_auxilliary(self, user_id, limit, offset, _seed, starting_point):
        if starting_point is None:

            # content found according to engagement metrics
            results_engagement = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter_by(
                    engagement_type=EngagementType.Like,
                )
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            list_engagement = list(map(lambda x: x[0], results_engagement))
            # get content most engaged with by user
            results_ANN = []

            # for likes

            table_ann_likes = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter_by(
                    user_id=user_id,
                    engagement_type=EngagementType.Like,
                    engagement_value=1,
                )
                # .where(Engagement.engagement_value==1)
                # .order_by(Engagement.engagement_value.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            # for engagement time

            table_ann_engagement_time = (
                Engagement.query.with_entities(Engagement.content_id, func.count())
                .filter_by(
                    user_id=user_id,
                    engagement_type=EngagementType.MillisecondsEngagedWith,
                )
                .order_by(Engagement.engagement_value.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            liked_by_user = list(map(lambda x: x[0], table_ann_likes)) + list(
                map(lambda x: x[0], table_ann_engagement_time)
            )

            # get ANN for the items found

            for liked_content in liked_by_user:
                content_ids, scores = ann_with_offset(
                    liked_content["content_id"],
                    0.9,
                    limit,
                    offset,
                    return_distances=True,
                )
            results_ANN.append(content_ids)

            return list_engagement + results_ANN, None
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
