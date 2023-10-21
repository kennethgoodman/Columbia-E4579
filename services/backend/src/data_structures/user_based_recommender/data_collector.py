
from src import db
import pandas as pd
from src.api.engagement.models import Engagement
import copy
from sqlalchemy import text, func, over, and_
from sqlalchemy.sql import alias
from sqlalchemy.sql.expression import bindparam


class DataCollector:
    _instance = None  # Singleton instance reference
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCollector, cls).__new__(cls)
            cls._instance.gather_data()
        return cls._instance

    def gather_data(self):
        random_order_cte = (
            db.session.query(
                Engagement.content_id,
                Engagement.user_id,
                Engagement.engagement_type,
                Engagement.engagement_value,
                over(
                    func.row_number(),
                    partition_by=Engagement.user_id,
                    order_by=text("(RAND())")
                ).label('rn')
            )
            .filter(
                Engagement.user_id >= 1  # first user of Fall 2023
            )
        ).cte()
        self.result = (
            db.session.query(
                random_order_cte.c.content_id,
                random_order_cte.c.user_id,
                random_order_cte.c.engagement_type,
                random_order_cte.c.engagement_value
            )
            .filter(
                text("rn <= 2000")  # get a max of 2k records per user
            )
            .order_by(
                random_order_cte.c.user_id,
                text("rn")
            )
        ).all()

    def get_data(self):
    	return copy.deepcopy(self.result)

    def get_data_df(self):
    	return pd.DataFrame(self.get_data(), columns=[
            'content_id', 'user_id', 'engagement_type', 'engagement_value'
        ])
