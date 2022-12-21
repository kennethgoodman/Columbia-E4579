from src.api.engagement.crud import get_dislike_count_by_content_id, get_like_count_by_content_id
from sqlalchemy.sql import text
from .AbstractFilter import AbstractFilter
from src import db

class EchoFilter(AbstractFilter):

    def filter_ids(self, user_id, content_ids, seed=None, starting_point=None):

        # Get contents that the user have already seen (will be fitered out)
        sql_statement = text(f"""
            		SELECT DISTINCT content_id 
            		FROM engagement 
            		WHERE user_id = {user_id}
            	""")
        with db.engine.connect() as con:
            seen = list(con.execute(sql_statement))
            seen = set(map(lambda x: x[0], seen))

        filtered = []
        for c_id in content_ids:
            like = get_like_count_by_content_id(c_id)
            dislike = get_dislike_count_by_content_id(c_id)

            if (dislike + like) == 0:
                like_ratio = 1
            else:
                like_ratio = like / (dislike + like)

            if (like_ratio >= 0.8) and (c_id not in seen):
                filtered.append(c_id)

        return filtered
