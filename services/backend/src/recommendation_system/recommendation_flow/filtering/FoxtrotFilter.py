from sqlalchemy.sql import text
from src import db
import pandas as pd
import numpy as np
from src.recommendation_system.recommendation_flow.filtering.AbstractFilter import (
    AbstractFilter,
)

df_user_clusters_like = pd.read_csv(
    r"/usr/src/app/src/foxtrot/foxtrot_users_clusters2.csv", nrows=100
)


class FoxtrotFilter(AbstractFilter):
    def filter_ids(self, content_ids, user_id, seed, starting_point):
        return content_ids
