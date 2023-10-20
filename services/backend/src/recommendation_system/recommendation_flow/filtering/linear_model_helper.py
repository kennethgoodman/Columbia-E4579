
from src import db
import pandas as pd
import copy
from sqlalchemy import text, func, over, and_
from sqlalchemy.sql import alias
from sqlalchemy.sql.expression import bindparam
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from flask import current_app
import traceback
from typing import List
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def fetch_data_stub_all_data():
    try:
        return db.session.query(
            Engagement.content_id,
            Engagement.user_id,
            Engagement.engagement_type,
            Engagement.engagement_value,
            GeneratedContentMetadata.guidance_scale,
            GeneratedContentMetadata.num_inference_steps,
            GeneratedContentMetadata.artist_style,
            GeneratedContentMetadata.source,
            GeneratedContentMetadata.model_version,
        ).join(
            GeneratedContentMetadata, Engagement.content_id == GeneratedContentMetadata.content_id
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

class DataCollector:
    _instance = None  # Singleton instance reference
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataCollector, cls).__new__(cls)
            cls._instance.gather_data()
        return cls._instance

    def artist_styles_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list is one larger to account for 'other'"
        )

    def sources_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list is one larger to account for 'other'"
        )

    def num_inference_steps_one_hot(self):
        raise NotImplementedError(
            "you need to implement this, needs to be two lists, one for string one for coefficient, coefficient list is one larger to account for 'other'"
        )

    def one_hot_encoding_functions(self):
        return zip(
            [self.artist_styles_one_hot(), self.sources_one_hot(), self.num_inference_steps_one_hot()],
            ['artist_style', 'source', 'num_inference_steps']
        )

    def feature_generation_user(self):
        user_features = self.user_data.groupby('user_id').agg({
            'user_likes': lambda x: np.sum((x['engagement_type'] == 'Like') & (x['engagement_value'] == 1)),
            'user_dislikes': lambda x: np.sum((x['engagement_type'] == 'Like') & (x['engagement_value'] == -1)),
            'user_engagement_time_avg': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].mean(),
            'user_engagement_time_median': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].median(),
            'user_engagement_time_p10': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].quantile(0.1),
            'user_engagement_time_p90': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].quantile(0.9)
        })
        self.user_data = pd.merge(self.user_data, user_features, on='user_id')

    def feature_generation_content_one_hot_encoding(self):
        self.content_data
        for (categories, _coefficient), col_name in self.one_hot_encoding_functions():
            self.content_data[col_name] = self.content_data[col_name].apply(lambda x: x if x in categories else 'other')
            encoder = OneHotEncoder(categories=categories + ['other'], sparse=False).fit_transform(
                self.content_data[[col_name]]
            )
            encoded_df = pd.DataFrame(encoded_styles, columns=encoder.get_feature_names_out([col_name]))
            self.content_data = pd.merge(self.content_data, encoded_df, left_index=True, right_index=True)

    def feature_generation_content_engagement_value(self):
        candidate_features = content_data.groupby('content_id').agg({
            'content_likes': lambda x: np.sum((x['engagement_type'] == 'Like') & (x['engagement_value'] == 1)),
            'content_dislikes': lambda x: np.sum((x['engagement_type'] == 'Like') & (x['engagement_value'] == -1)),
            'content_engagement_time_avg': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].mean(),
            'content_engagement_time_median': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].median(),
            'content_engagement_time_p10': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].quantile(0.1),
            'content_engagement_time_p90': lambda x: x[x['engagement_type'] == 'MillisecondEngageTime']['engagement_value'].quantile(0.9)
        })
        candidate_features.columns = ['_'.join(col).strip() for col in candidate_features.columns.values]
        self.content_data = pd.merge(
            self.content_data,
            candidate_features,
            on="content_id"
        )

    def feature_generation_content(self):
        self.feature_generation_content_engagement_value()
        return self.content_data.groupby('content_id').agg(lambda x: x.iloc[0]) # take first value, all should be same

    def gather_data(self, user_id, content_ids):
        columns = [
            'content_id',
            'user_id',
            'engagement_type',
            'engagement_value',
            'seed',
            'guidance_scale',
            'num_inference_steps',
            'artist_style',
            'source',
            'model_version'
        ]
        self.content_data = pd.DataFrame(fetch_data_stub().filter(
            Content.id in content_ids
        ).all(), columns=columns)
        self.user_data = pd.DataFrame(fetch_data_stub().filter(
            Engagement.user_id == user_id
        ).all(), columns=columns)
        
    def threshold(self):
        raise NotImplementedError("you need to implement")

    def bias(self):
        raise NotImplementedError("you need to implement")

    def coefficients(self):
        return {
            'content_likes': 0.0,
            'content_dislikes': 0.0,
            'content_engagement_time_avg': 0.0,
            'content_engagement_time_median': 0.0,
            'content_engagement_time_p10': 0.0,
            'content_engagement_time_p90': 0.0,

            'user_likes': 0.0,
            'user_dislikes': 0.0,
            'user_engagement_time_avg': 0.0,
            'user_engagement_time_median': 0.0,
            'user_engagement_time_p10': 0.0,
            'user_engagement_time_p90': 0.0,
        }

    def run_linear_model(self):
        coeffs = self.coefficients()
        for (categories, _coefficient), col_name in self.one_hot_encoding_functions():
            coeffs[col_name] = _coefficient

        self.results['linear_output'] = self.bias()
        for col_name, _coefficient in coeffs.items():
            self.results['linear_output'] += self.results[col_name] * _coefficient
        return self.results[self.results['linear_output'] >= threshold]['content_ids']