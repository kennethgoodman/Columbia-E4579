import operator
from sqlalchemy.sql.expression import func
from src import db
from sqlalchemy.sql import text
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator


class BetaGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if starting_point is None:
            #1. generate based on like number
            results_like = (
                Engagement.query.with_entities(
                    Engagement.content_id, func.count()
                )
                .filter_by(
                    engagement_type=EngagementType.Like,
                )
                .group_by(Engagement.content_id)
                .order_by(func.count().desc())
                # .limit(int(limit * 0.4))
                .limit(int(limit))
                .offset(offset)
                .all()
            )
            #print(type(results)) -- list
            #2. generate based on randomness
            results_random = (
                Content.query.with_entities(Content.id,1)
                .order_by(func.random(seed))
                .limit(int(limit * 0.2))
                .offset(offset)
                .all()
            )
          
            #3. generate based on engagement time
            results_engage_time = (
               Engagement.query.with_entities(Engagement.content_id,2,Engagement.engagement_value)
               .filter(Engagement.engagement_value <= 20000) 
               .filter(Engagement.engagement_value >= 1000) 
               .order_by(Engagement.engagement_value.desc())
               # .limit(int(limit * 0.4))
               .limit(int(limit))
               .offset(offset)
               .all()
            )

            #4. generate based on user liked source/style
            get_content_ids_user_liked_source = text(f"""
                select content_id 
                from generated_content_metadata 
                where source IN 
                (select distinct source from engagement e JOIN generated_content_metadata g 
                ON e.content_id = g.content_id 
                where user_id = {user_id} and engagement_type = 'Like' and engagement_value = 1) and 
                content_id NOT IN (select content_id from engagement 
                where user_id = {user_id} and engagement_type = 'Like' and engagement_value = 1)
                LIMIT {str(limit * 0.2).split('.')[0]}
                OFFSET {offset};
            """)

            with db.engine.connect() as con:
                content_ids_user_liked_source = list(con.execute(get_content_ids_user_liked_source))

            content_ids_user_liked_source = [tuple(i) + (1,) for i in content_ids_user_liked_source] # make it tuple for score
            
            results = list(set(results_like + results_random + results_engage_time + content_ids_user_liked_source))
            return list(map(lambda x: x[0], results)), list(map(lambda x: x[1], results)) #number of like as score
        elif starting_point.get("content_id", False):
            content_ids, scores = ann_with_offset(
                starting_point["content_id"], 0.9, limit, offset, return_distances=True
            )
            return content_ids, scores
        raise NotImplementedError("Need to provide a key we know about")
