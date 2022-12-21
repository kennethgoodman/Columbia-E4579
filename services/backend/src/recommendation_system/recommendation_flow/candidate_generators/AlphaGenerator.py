from sqlalchemy import text
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.data_structures.approximate_nearest_neighbor import ann_with_offset
from .AbstractGenerator import AbstractGenerator
from src.api.utils.auth_utils import get_user
import json

class AlphaGenerator(AbstractGenerator):
    def get_content_ids(self, user_id):
        print("user id from alpha generator:",user_id)
        # all images that have not been engaged by this user can be a candidate
        sql_statement_candidates = f"""
        SELECT id 
        FROM content 
        WHERE id NOT IN (
            SELECT content_id FROM engagement WHERE user_id={user_id}
        );
    	"""
        with db.engine.connect() as con:
            candidates = list(con.execute(sql_statement_candidates))
        candidates = [i[0] for i in candidates]
        # print('candidates from sql:',candidates)
        
        return candidates, None
