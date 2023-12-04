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

with open(legalize('foxtrot_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)


with open(legalize('foxtrot_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)


class FoxtrotFeatureGeneration(AbstractFeatureGeneration):
    def custom_aggregation(self, prefix, data):
        result = {
            f'{prefix}_likes': np.sum((data['engagement_type'] == 'Like') & (data['engagement_value'] == 1)),
            f'{prefix}_dislikes': np.sum((data['engagement_type'] == 'Like') & (data['engagement_value'] == -1)),
            f'{prefix}_engagement_time_avg': data[data['engagement_type'] == 'MillisecondsEngagedWith'][
                'engagement_value'].mean(),
        }
        return pd.Series(result)

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

        feature_df = self.user_data.groupby('user_id').apply(
            lambda data: self.custom_aggregation('user', data)).reset_index()
        feature_df = feature_df.fillna(0)

        return feature_df, ['user_likes', 'user_dislikes', 'user_engagement_time_avg'], []

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

        overall = zip(
            [
                ['medieval', 'anime', 'studio', 'oil_on_canvas', 'unreal_engine', 'edward_hopper', 'shepard_fairey'],
                ['human_prompts', 'r/EarthPorn', 'r/Showerthoughts', 'r/scifi', 'r/pics', 'r/educationalgifs',
                 'r/Damnthatsinteresting'],
                ['20', '50', '100'],
            ],
            ['artist_style', 'source', 'num_inference_steps']
        )

        feature_df = self.generated_content_metadata.copy()
        feature_df['num_inference_steps'] = feature_df['num_inference_steps'].astype(str)
        for categories, col_name in overall:
            feature_df[col_name] = feature_df[col_name].apply(lambda x: x if x in categories else 'other').to_frame()
        content_nums = self.engagement_data.groupby('content_id').apply(
            lambda data: self.custom_aggregation('content', data)).reset_index()
        feature_df = pd.merge(
            feature_df,
            content_nums,
            on='content_id',
            how='left'
        ).fillna(0)

        return feature_df, ['guidance_scale', 'content_likes', 'content_dislikes', 'content_engagement_time_avg'], [
            'artist_style', 'source', 'num_inference_steps']

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

class FoxtrotModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = kwargs['fg'].X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "FoxtrotModel"
