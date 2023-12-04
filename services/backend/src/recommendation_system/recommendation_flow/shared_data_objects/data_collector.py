from src import db
import pandas as pd
import copy
from sqlalchemy import text, func, over, and_, cast, String
from sqlalchemy.sql import alias
from sqlalchemy.sql.expression import bindparam
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from flask import current_app
import traceback
from typing import List
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def fetch_engagement_data(_filter, n_rows_per_content):
    cte = db.session.query(
        Engagement.content_id,
        Engagement.user_id,
        cast(Engagement.engagement_type, String).label('engagement_type'),
        Engagement.engagement_value,
        func.row_number().over(
            partition_by=Engagement.content_id,
            order_by=Engagement.user_id  # Adjust the ordering as per your requirement
        ).label('row_num')
    ).filter(
        _filter
    ).cte()

    cte_alias = alias(cte, name='cte_alias')

    try:
        return pd.DataFrame(
            db.session.query(cte_alias).filter(cte_alias.c.row_num <= n_rows_per_content).all(),
            columns=[
                'content_id', 'user_id', 'engagement_type', 'engagement_value', 'row_num'
            ]
        )
    except Exception as e:
        print(f"Error fetching engagement data: {e}")
        print(traceback.format_exc())
        return None


def fetch_generated_content_metadata_data(content_ids):
    try:
        return pd.DataFrame(
            db.session.query(
                GeneratedContentMetadata.content_id,
                GeneratedContentMetadata.guidance_scale,
                GeneratedContentMetadata.num_inference_steps,
                GeneratedContentMetadata.artist_style,
                GeneratedContentMetadata.source,
            ).filter(
                GeneratedContentMetadata.content_id.in_(content_ids)
            ).all(),
            columns=[
                'content_id',
                'guidance_scale',
                'num_inference_steps',
                'artist_style',
                'source'
            ]
        )
    except Exception as e:
        print(f"Error fetching generated content metadata data: {e}")
        print(traceback.format_exc())
        return None


class DataCollector:
    def get_engagement_data(self, content_ids):
        return fetch_engagement_data(
            Engagement.content_id.in_(content_ids),
            50
        )

    def get_generated_content_metadata_data(self, content_ids):
        return fetch_generated_content_metadata_data(
            content_ids
        )

    def get_user_data(self, user_id):
        return fetch_engagement_data(
            Engagement.user_id == user_id,
            500
        )

    def gather_data(self, user_id, content_ids):
        self.engagement_data = self.get_engagement_data(content_ids)
        self.generated_content_metadata_data = self.get_generated_content_metadata_data(content_ids)
        self.user_data = self.get_user_data(user_id)

    def return_data_copy(self):
        return (
            copy.deepcopy(self.engagement_data),
            copy.deepcopy(self.generated_content_metadata_data),
            copy.deepcopy(self.user_data)
        )
