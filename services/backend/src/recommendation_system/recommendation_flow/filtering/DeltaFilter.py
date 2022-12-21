import random
from sqlalchemy.sql import text
from src import db
import csv
import os
import time

from .AbstractFilter import AbstractFilter


class QualityFilter(AbstractFilter):
    def filter_ids(self, content_ids, _, starting_point, user_id):

        ### 1. Remove images that user disliked previously
        def remove_user_dislikes():
            sql_statement = text(f"""
                -- 1. User dislike
                SELECT engagement.content_id
                FROM engagement 
                    WHERE engagement_type = 'Like' 
                    and user_id = {user_id}
                    and engagement_value = -1 
                GROUP BY 1
            """)
            with db.engine.connect() as con:
                ids_to_filter_out = list(con.execute(sql_statement))
                ids_to_filter_out = set(map(lambda x: x[0], ids_to_filter_out))
            return ids_to_filter_out

        ### 2. Remove images from artist style that user disliked previously
        def remove_user_artist_style_dislikes():
            sql_statement = text(f"""
                -- 2. User group by artist_style dislike
                with dislike_user as (
                SELECT generated_content_metadata.artist_style,
                    SUM(CASE WHEN engagement_value = -1 THEN 1 ELSE 0 END) as dislike,
                    COUNT(*) as total_engagement,
                    SUM(CASE WHEN engagement_value = -1 THEN 1 ELSE 0 END)/COUNT(*) as dislike_ratio
                FROM engagement left join generated_content_metadata on  engagement.content_id = generated_content_metadata.content_id
                WHERE engagement_type = 'Like'
                    and user_id = {user_id}
                GROUP BY generated_content_metadata.artist_style
                )

                SELECT engagement.content_id
                FROM engagement 
                    LEFT JOIN generated_content_metadata ON engagement.content_id = generated_content_metadata.content_id 
                    LEFT JOIN dislike_user ON generated_content_metadata.artist_style = dislike_user.artist_style
                WHERE dislike_user.dislike_ratio > 0.5 and dislike_user.dislike > 3
                GROUP BY 1
            """)
            with db.engine.connect() as con:
                ids_to_filter_out = list(con.execute(sql_statement))
                ids_to_filter_out = set(map(lambda x: x[0], ids_to_filter_out))
            return ids_to_filter_out

        ### 3. Remove images that have high dislike/like ratio
        def remove_overall_high_dislike_to_like_ratio():
            sql_statement = text(f"""
                with dislike_overall as(
                SELECT generated_content_metadata.artist_style,
                    SUM(CASE WHEN engagement_value = -1 THEN 1 ELSE 0 END) as dislike,
                    COUNT(*) as total_engagement,
                    SUM(CASE WHEN engagement_value = -1 THEN 1 ELSE 0 END)/COUNT(*) as dislike_ratio
                FROM engagement left join generated_content_metadata on  engagement.content_id = generated_content_metadata.content_id
                WHERE engagement_type = 'Like'
                GROUP BY generated_content_metadata.artist_style
                )

                SELECT engagement.content_id
                FROM engagement 
                    LEFT JOIN generated_content_metadata ON engagement.content_id = generated_content_metadata.content_id 
                    LEFT JOIN dislike_overall ON generated_content_metadata.artist_style = dislike_overall.artist_style
                WHERE dislike_overall.dislike_ratio > 0.5 and dislike_overall.dislike > 10
                GROUP BY 1
            """)
            with db.engine.connect() as con:
                ids_to_filter_out = list(con.execute(sql_statement))
                ids_to_filter_out = set(map(lambda x: x[0], ids_to_filter_out))
            return ids_to_filter_out

        ### 4. Show only High Quality Images
        def remove_low_quality_images(content_ids):
            content_ids_string = list(map(lambda x: str(x), content_ids))
            content_isin_string = "(" + ','.join(content_ids_string) + ")"

            with db.engine.connect() as conn:
                scores = conn.execute(f"SELECT * from score "\
                        f"where content_id IN "\
                        f"{content_isin_string}").all()

            scores_sorted = list(sorted(scores, key = lambda x : float(x[1])))
            scores_filtered = scores_sorted[:int(0.90*len(scores_sorted))]

            return dict(scores_filtered)

        start = time.time()
        ids_to_filter_out1 = remove_user_dislikes()
        print(f'FILTERING: Stage 1 no. of images to filter: {len(ids_to_filter_out1)}, Time: {(time.time() - start)}')

        scores_filtered = remove_low_quality_images(content_ids)
        ids_to_filter_out4 = set(scores_filtered.keys())

        print(f'FILTERING: Stage 4 no. of images to filter: {len(ids_to_filter_out4)}, Time: {(time.time() - start)}')

        total_ids_to_drop = set.union(ids_to_filter_out1,
                                      ids_to_filter_out4)

        filtered_content_ids = set.difference(set(content_ids),
                                                  total_ids_to_drop)

        print(f'FILTERING: no. of images (before): {len(content_ids)}')
        print(f'FILTERING: no. of images (after): {len(filtered_content_ids)}')

        return scores_filtered

        # return filtered_content_ids
