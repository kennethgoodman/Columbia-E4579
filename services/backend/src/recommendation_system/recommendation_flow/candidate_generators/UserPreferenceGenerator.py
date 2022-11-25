import operator
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator


class UserPreferenceGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):

        if starting_point is None:
            # get the preference from the user
            with db.engine.connect() as con:
                query = """select
                                content_id
                            from
                            (
                                    select
                                        engagement.content_id,
                                        sum(engagement.engagement_value)
                                    from
                                        (
                                            select
                                                content_id
                                            from
                                                generated_content_metadata
                                                left join (
                                                    select
                                                        distinct temp1.source
                                                    from
                                                        (
                                                            select
                                                                generated_content_metadata.source
                                                            from
                                                                (
                                                                    select
                                                                        distinct engagement.content_id as content_id
                                                                    from
                                                                        engagement
                                                                    where
                                                                        engagement.user_id = """ + str(user_id) + """
                                                                        and (
                                                                            engagement.engagement_type = "like"
                                                                            and engagement.engagement_value = 1
                                                                        )
                                                                ) temp
                                                                left join engagement on engagement.content_id = temp.content_id
                                                                left join generated_content_metadata on engagement.content_id = generated_content_metadata.content_id
                                                            group by
                                                                engagement.content_id,
                                                                generated_content_metadata.source
                                                            having
                                                                sum(engagement.engagement_value) < 600000
                                                            order by
                                                                sum(engagement.engagement_value) desc
                                                            limit
                                                                10
                                                        ) temp1
                                                    order by
                                                        RAND ()
                                                    limit
                                                        1
                                                ) temp2 on temp2.source = generated_content_metadata.source
                                        ) temp3
                                        left join engagement on engagement.content_id = temp3.content_id
                                    group by
                                        engagement.content_id
                                    having
                                        sum(engagement.engagement_value) < 600000
                                        and sum(engagement.engagement_value) > 6000
                                    order by
                                        sum(engagement.engagement_value) desc
                                ) temp4
                            order by
                                RAND ()
                            limit
                                """ + str(limit) + """;
                            """

                results = con.execute(query).all()

            num_results = len(results)

            # if no previous record then do random cg
            if num_results == 0:
                return RandomGenerator().get_content_ids(
                    user_id, limit, offset, seed, starting_point
                )

            ids = list(map(lambda x: x[0], results))

            return ids, None

        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
