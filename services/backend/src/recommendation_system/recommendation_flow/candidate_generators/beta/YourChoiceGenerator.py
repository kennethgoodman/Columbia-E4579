
from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
#from .AbstractGenerator import AbstractGenerator
from src.api.utils.auth_utils import get_user
import json
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator

class YourChoiceGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):
        sql_statement_candidates = text(f"""
        SELECT content_id 
        FROM engagement
        WHERE engagement_type = 'Like' AND engagement_value = 1
        GROUP BY content_id
        ORDER BY COUNT(content_id) DESC
        LIMIT {limit}
        OFFSET {offset};
        """)
        with db.engine.connect() as con:
            candidates = list(con.execute(sql_statement_candidates))

        candidates = [i[0] for i in candidates]
        
        return candidates, candidates
    
    def _get_name(self):
        return "YourChoiceGenerator"
