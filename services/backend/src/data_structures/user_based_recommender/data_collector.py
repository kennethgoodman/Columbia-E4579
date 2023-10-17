
from src import db
import pandas as pd
from src.api.engagement.models import Engagement
import copy
from sqlalchemy import or_, and_, func
from sqlalchemy.sql.expression import bindparam


class DataCollector:
    _instance = None  # Singleton instance reference
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCollector, cls).__new__(cls)
            cls._instance.gather_data()
        return cls._instance

    def gather_data(self):
        self.result = (
            session.query(
                Engagement.content_id,
                Engagement.user_id,
                Engagement.engagement_type,
                Engagement.engagement_value
            )
            .filter(
                Engagement.user_id >= 77 # first user of Fall 2023
            )
            .from_self()
            .add_columns(
                over(
                    func.row_number(),
                    partition_by=Engagement.user_id,
                    order_by=text("(RAND())")
                ).label('rn')
            )
            .filter(text("rn <= 2000")) # get a max of 2k records per user
            .order_by(Engagement.user_id, text("rn"))
        ).all()

    def get_data(self):
    	return copy.deepcopy(self.result)

    def get_data_df(self):
    	return pd.DataFrame(self.get_data(), columns=[
            'content_id', 'user_id', 'engagement_type', 'engagement_value'
        ])
