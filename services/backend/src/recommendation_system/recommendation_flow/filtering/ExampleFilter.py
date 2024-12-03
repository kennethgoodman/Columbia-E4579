from sqlalchemy.sql import text
from src import db
from src.api.engagement.models import Engagement, EngagementType

from .AbstractFilter import AbstractFilter


class ExampleFilter(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, _, starting_point, amount=None, dc=None):
        sql_statement = text(f"""
            SELECT content_id, 
                SUM(CASE WHEN engagement_value = 1 THEN 1 ELSE 0 END) as likes, 
                SUM(CASE WHEN engagement_value = -1 THEN 1 ELSE 0 END) as dislikes,
                count(*)
            FROM engagement
            WHERE engagement_type = 'Like'
                and content_id in ({','.join(map(str, content_ids))})
            GROUP BY content_id
        """)
        ids_to_filter_out = []
        with db.engine.connect() as con:
            ids_to_filter_out_ = list(con.execute(sql_statement))
            line = sorted(
                map(
                    lambda x: x[1] / max(x[2], 1), 
                    ids_to_filter_out_
                )
            )[
                len(ids_to_filter_out_)//4*3
            ] # top 75%
            ids_to_filter_out = set(
                map(lambda x: x[0],
                    filter(
                        lambda x: x[1] / max(x[2], 1) < line,
                        ids_to_filter_out_
                    )
                )
            )
        filtered_content_ids = []
        for content_id in content_ids:
            if content_id in ids_to_filter_out:
                continue
            filtered_content_ids.append(content_id)
        return filtered_content_ids

    def _get_name(self):
        return "ExampleFilter"


class ExampleFilterWithExploration(AbstractFilter):
    def _filter_ids(self, user_id, content_ids, _, starting_point, amount=None, dc=None):
        sql_statement = text(f"""
            SELECT content_id, 
                SUM(CASE WHEN engagement_value = 1 THEN 1 ELSE 0 END) as likes, 
                SUM(CASE WHEN engagement_value = -1 THEN 1 ELSE 0 END) as dislikes,
                count(*)
            FROM engagement
            WHERE engagement_type = 'Like'
                and content_id in ({','.join(map(str, content_ids))})
            GROUP BY content_id
        """)
        engaged_content_ids = Engagement.query.with_entities(Engagement.content_id).filter(
              Engagement.user_id == user_id, 
              Engagement.content_id.in_(content_ids),
              Engagement.engagement_type == EngagementType.Like
        ).distinct().all()
        import random
        ids_to_filter_out = []
        with db.engine.connect() as con:
            ids_to_filter_out_ = list(con.execute(sql_statement))
            line = sorted(
                map(
                    lambda x: x[1] / max(x[2], 1), 
                    ids_to_filter_out_
                )
            )[
                len(ids_to_filter_out_)//4*3
            ] # top 75%
            ids_to_filter_out = set(
                map(lambda x: x[0],
                    filter(
                        lambda x: x[1] / max(x[2], 1) < line,
                        ids_to_filter_out_
                    )
                )
            )
        filtered_content_ids = []
        for content_id in content_ids:
            if content_id in engaged_content_ids:
                continue
            # if content_id in ids_to_filter_out and random.random() > 0.5:
                # continue
            filtered_content_ids.append(content_id)
        return filtered_content_ids

    def _get_name(self):
        return "ExampleFilterWithExploration"
