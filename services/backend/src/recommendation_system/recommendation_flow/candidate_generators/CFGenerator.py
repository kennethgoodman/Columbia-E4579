import operator
from sqlalchemy.sql.expression import func
from src import db
from src.api.content.models import Content
from src.api.engagement.models import Engagement, EngagementType
from src.data_structures.approximate_nearest_neighbor import ann_with_offset

from .AbstractGenerator import AbstractGenerator
from .RandomGenerator import RandomGenerator

class CFGenerator(AbstractGenerator):
    def get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        with db.engine.connect() as con:
            prefs = con.execute(f'SELECT prefs from user_prefs where id = {user_id}').all()[0][0]

            # return empty if no prefs
            if prefs == "":
                return [], None

            # retrieve all pictures from preference categories
            query = 'SELECT content.id as id ' \
                    'from content left outer join ' \
                    'generated_content_metadata ' \
                    'on content.id = generated_content_metadata.content_id ' \
                    f'where generated_content_metadata.artist_style IN({prefs[1:-1]});'

            ids = list(map(lambda x: x[0], con.execute(query).all()))

            return ids, None
