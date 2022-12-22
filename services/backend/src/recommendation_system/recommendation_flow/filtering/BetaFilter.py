import random
from sqlalchemy.sql import text
from src import db

from .AbstractFilter import AbstractFilter


class BetaFilter(AbstractFilter):
    def filter_ids(self, content_ids, user_id, _, starting_point, shown):
        # filter out images with dislike >= 1
        sql_statement = text(f"""
        	SELECT content_id 
        	FROM engagement 
        	WHERE engagement_type = 'Like' 
        		and engagement_value = -1 -- dislikes
        		and content_id in ({','.join(map(str, content_ids))})
        	GROUP BY content_id
        """)

        with db.engine.connect() as con:
            ids_to_filter_out = list(con.execute(sql_statement)) # [(id1,), (id2,)...]
            ids_to_filter_out = set(map(lambda x: x[0], ids_to_filter_out)) # {id1, id2, ...}
            
        filtered_content_ids = []
        for content_id in content_ids:
            if content_id in ids_to_filter_out:
                continue
            if content_id in shown:
                continue
            filtered_content_ids.append(content_id)
        return filtered_content_ids