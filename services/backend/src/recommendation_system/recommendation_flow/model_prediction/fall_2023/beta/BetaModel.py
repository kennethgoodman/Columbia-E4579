from typing import Tuple, List
import pandas as pd
from datetime import datetime
import numpy as np
import os
import pickle
from src.recommendation_system.recommendation_flow.model_prediction.postprocess import (
    Postprocessor,
    AbstractFeatureGeneration
)
from src.recommendation_system.recommendation_flow.model_prediction.AbstractModel import AbstractModel

script_dir = os.path.dirname(os.path.abspath(__file__))
legalize = lambda s: os.path.join(script_dir, s)

from sklearn.ensemble import RandomForestRegressor
class DummyModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=50)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
import sys
sys.modules['__main__'].DummyModel = DummyModel
with open(legalize('beta_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
del sys.modules['__main__'].DummyModel


with open(legalize('beta_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)

class BetaFeatureGeneration(AbstractFeatureGeneration):
    def feature_generation_user(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Generates user features. Keep all the categorical variables as is,
        since the one-hot encoding will be done by our own pipeline. Along with
        the feature dataframe, you'll need to output lists of numberical features
        and categorical features as well.

        Returns
          pd.DataFrame: User feature dataframe
          List[str]: List of numerical features. E.g. ['feat_1', 'feat_3, ...]
          List[str]: List of categorical features. E.g. ['feat_2', 'feat_4, ...]
        """

        df = self.user_data.copy()
        # Filter rows where engagement_type is 'Like'
        like_df = df[(df['engagement_type'] == 'Like') & (df['engagement_value'] == 1)]
        dislike_df = df[(df['engagement_type'] == 'Like') & (df['engagement_value'] == -1)]
        like_df = like_df.groupby('user_id').size().reset_index(name='like_count')
        dislike_df = dislike_df.groupby('user_id').size().reset_index(name='dislike_count')
        merged_df = pd.merge(like_df, dislike_df, on='user_id')
        merged_df['like-dislike'] = merged_df['like_count'] - merged_df['dislike_count']

        # Filter rows where engagement_type is 'MillisecondsEngagedWith'
        engaged_df = df[df['engagement_type'] == 'MillisecondsEngagedWith']
        avg_engagement_values = engaged_df.groupby('user_id')['engagement_value'].mean().reset_index(name='avg_engagement_value')

        # Merge the two results based on user_id
        feature_df = pd.merge(merged_df, avg_engagement_values, on='user_id', how='outer').fillna(0)

        return feature_df, ['like_count', 'dislike_count', 'like-dislike', 'avg_engagement_value'], []

    def feature_generation_content(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Generates content features. Keep all the categorical variables as is,
        since the one-hot encoding will be done by our own pipeline. Along with
        the feature dataframe, you'll need to output lists of numberical features
        and categorical features as well.

        Returns
          pd.DataFrame: User feature dataframe
          List[str]: List of numerical features. E.g. ['feat_1', 'feat_3, ...]
          List[str]: List of categorical features. E.g. ['feat_2', 'feat_4, ...]
        """

        artist_styles_categories = ['van_gogh', 'jean-michel_basquiat', 'detailed_portrait', 'kerry_james_marshall', 'medieval']
        sources_categories = ['human_prompts', 'r/Showerthoughts', 'r/EarthPorn', 'r/scifi', 'r/pics']
        feature_df = self.generated_content_metadata.copy()
        feature_df['artist_style'] = feature_df['artist_style'].apply(lambda x: x if (x and x in artist_styles_categories) else 'other')
        feature_df['source'] = feature_df['source'].apply(lambda x: x if (x and x in sources_categories) else 'other')

        return feature_df, ['guidance_scale', 'num_inference_steps'], ['artist_style', 'source']

    def load_postprocessor(self):
        return POST_PROCESSOR

    def predict_probabilities(self, X) -> Tuple[list, list, list, list]:
        """Predicts the 3 target variables by using the model that you trained.
        Make sure you load the model properly.

        Args:
            X (pd.DataFrame): Feature dataframe with 2-level index of (user_id, content_id)

        Returns:
            (list, list, list): (predicted prbability of like,
                                 predicted probability of dislike,
                                 predicted engagement time)
        """
        pred_like = MODEL['like'].predict(X)
        pred_dislike = MODEL['dislike'].predict(X)
        pred_engtime = MODEL['engage_time'].predict(X)
        return pred_like, pred_dislike, pred_engtime, X.index.values


class BetaModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = kwargs['fg'].X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "BetaModel"
