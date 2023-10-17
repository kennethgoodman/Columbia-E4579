
from src import db
import pandas as pd
from src.api.engagement.models import Engagement
import copy

class DataCollector:
    _instance = None  # Singleton instance reference
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCollector, cls).__new__(cls)
            cls._instance.gather_data()
        return cls._instance

    def gather_data(self):
    	self.result = db.session.query(
            Engagement.content_id,
            Engagement.user_id,
            Engagement.engagement_type,
            Engagement.engagement_value
        ).all()

    def get_data(self):
    	return copy.deepcopy(self.result)

    def get_data_df(self):
    	return pd.DataFrame(self.get_data(), columns=[
            'content_id', 'user_id', 'engagement_type', 'engagement_value'
        ])
