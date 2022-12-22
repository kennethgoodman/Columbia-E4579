import random
# from sqlalchemy.sql import text
from src import db
from .AbstractFilter import AbstractFilter
import pickle
import pandas as pd
import csv

with open('/usr/src/app/src/alpha/offensive_blacklist.csv') as f:
    df_list = list(csv.reader(f))
    offensive_blacklist = list(map(lambda x: int(x[0]),df_list[1:]))

class AlphaFilter(AbstractFilter):
    def filter_ids(self, content_ids): # content_ids are the fresh candidates not in queue
        # general model: get popular and universally liked images
        # order by top liked, top engagemet, top engagement time, add add in random ones if need more output
        # return 1000 images
        # get top liked:
        sql_statement_top_liked = f"""
            SELECT content_id
            FROM engagement 
            WHERE engagement_type = 'Like' AND content_id NOT IN {tuple(offensive_blacklist)}
            GROUP BY content_id
            ORDER BY SUM(engagement_value)/SUM(ABS(engagement_value)) DESC;
    	"""  # limit ;
        with db.engine.connect() as con:
            filtered_content_ids_top_liked = list(con.execute(sql_statement_top_liked))
        filtered_content_ids_top_liked = [i[0] for i in filtered_content_ids_top_liked]
        filtered_content_ids_combined = filtered_content_ids_top_liked
        # get top engagement 750ms-3000ms engagement time:
        sql_statement_top_engagement_time = f"""
            SELECT content_id
            FROM engagement
            WHERE engagement_type = 'MillisecondsEngagedWith' AND engagement_value < 25000 AND content_id NOT IN {tuple(filtered_content_ids_combined)} AND content_id NOT IN {tuple(offensive_blacklist)}
            GROUP BY content_id
            ORDER BY SUM(engagement_value) DESC;
        """
        with db.engine.connect() as con:
            filtered_content_ids_top_engagement_time = list(con.execute(sql_statement_top_engagement_time))
        filtered_content_ids_top_engagement_time = [i[0] for i in filtered_content_ids_top_engagement_time]
        filtered_content_ids_combined.extend(filtered_content_ids_top_engagement_time)
        
        # get top engagements:
        sql_statement_top_engagement = f"""
    		SELECT content_id
            FROM engagement 
            WHERE engagement_type = 'Like' AND content_id NOT IN {tuple(filtered_content_ids_combined)} AND content_id NOT IN {tuple(offensive_blacklist)}
            GROUP BY content_id
            ORDER BY SUM(ABS(engagement_value)) DESC;
    	"""   # limit ;
        with db.engine.connect() as con:
            filtered_content_ids_top_engagement = list(con.execute(sql_statement_top_engagement))
        filtered_content_ids_top_engagement = [i[0] for i in filtered_content_ids_top_engagement]
        filtered_content_ids_combined.extend(filtered_content_ids_top_engagement)
        
        # if not enough results (<1000), include the rest of the candidates
        if len(filtered_content_ids_combined) < 1000:
            sql_statement_add_ons = f"""
                SELECT id
                FROM content
                WHERE id IN {tuple(content_ids)} AND id NOT IN {tuple(filtered_content_ids_combined)} AND id NOT IN {tuple(offensive_blacklist)}
                LIMIT {1000-len(filtered_content_ids_combined)};
            """
            with db.engine.connect() as con:
                content_ids_add_ons = list(con.execute(sql_statement_add_ons))
            content_ids_add_ons = [i[0] for i in content_ids_add_ons]
            filtered_content_ids_combined.extend(content_ids_add_ons)
        return filtered_content_ids_combined


