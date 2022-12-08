import random
# from sqlalchemy.sql import text
from src import db
from .AbstractFilter import AbstractFilter


class AlphaFilter(AbstractFilter):
    def filter_ids(self, content_ids, seed, starting_point): # content_ids are the fresh candidates not in queue

        # general model: get popular and universally liked images
        # order by total # likes + total # dislikes, and total # likes / (total # likes + total  dislikes)
        # get the top ___ images

        # what to do in cold start when there's no engagement? then put some random ones in
        # get top engagements:
        sql_statement_top_engagement = """
    		SELECT content_id
            FROM engagement 
            WHERE engagement_type = 'Like' 
            GROUP BY content_id
            ORDER BY SUM(ABS(engagement_value)) DESC
    	"""   # limit ;
        with db.engine.connect() as con:
            filtered_content_ids_top_engagement = list(con.execute(sql_statement_top_engagement))
        filtered_content_ids_top_engagement = [i[0] for i in filtered_content_ids_top_engagement]
        print('filtered_content_ids_top_engagement',filtered_content_ids_top_engagement)

        # get top liked:
        sql_statement_top_liked = """
            SELECT content_id
            FROM engagement 
            WHERE engagement_type = 'Like' 
            GROUP BY content_id
            ORDER BY SUM(engagement_value)/SUM(ABS(engagement_value)) DESC
    	"""  # limit ;
        with db.engine.connect() as con:
            filtered_content_ids_top_liked = list(con.execute(sql_statement_top_liked))
        filtered_content_ids_top_liked = [i[0] for i in filtered_content_ids_top_liked]
        print('filtered_content_ids_top_liked',filtered_content_ids_top_liked)
        

        # to combine top engagement and top liked, want more top liked on top, and more number of top liked 
        filtered_content_ids_combined = list(set(filtered_content_ids_top_engagement).union(set(filtered_content_ids_top_liked)))  # union of top engagement and top liked, ratio is 
        print('filtered_content_ids_combined',filtered_content_ids_combined)
        return filtered_content_ids_combined
