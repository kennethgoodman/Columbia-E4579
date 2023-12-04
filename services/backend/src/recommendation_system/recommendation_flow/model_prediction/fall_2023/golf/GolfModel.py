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

with open(legalize('golf_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)


with open(legalize('golf_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)


class GolfFeatureGeneration(AbstractFeatureGeneration):
    def feature_generation_user(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        # Group by user_id and aggregate features
        user_features = self.user_data.groupby('user_id').agg(
            categorical_feature=('engagement_type', lambda x: x.mode()[0]),
            numerical_feature=('engagement_value', 'mean'),
            latest_created_date=('created_date', 'max')
        ).reset_index()

        # Convert created_date to a numerical feature representing the time elapsed since the latest engagement
        user_features['latest_created_date'] = (datetime.now() - pd.to_datetime(user_features['latest_created_date'])).dt.total_seconds()

        # Rename columns for clarity
        user_features.rename(columns={
            'categorical_feature': 'user_categorical_feature',
            'numerical_feature': 'user_numerical_feature',
            'latest_created_date': 'user_time_elapsed'
        }, inplace=True)

        # Define lists of numerical and categorical features
        numerical_features = ['user_numerical_feature', 'user_time_elapsed']
        categorical_features = ['user_categorical_feature']

        return user_features, numerical_features, categorical_features

    def feature_generation_content(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        # Assuming self.generated_content_metadata contains the content metadata
        content_features = self.generated_content_metadata_data[['content_id', 'guidance_scale', 'num_inference_steps', 'source']].copy()

        # One-hot encode the 'source' column
        content_features = pd.get_dummies(content_features, columns=['source'], prefix='content_source')

        # Rename columns for clarity
        content_features.rename(columns={
            'guidance_scale': 'content_numerical_feature_guidance_scale',
            'num_inference_steps': 'content_numerical_feature_num_inference_steps',
            'content_source_human_prompts': 'content_source_human_prompts',
            'content_source_r/pics': 'content_source_r/pics',
            'content_source_r/scifi': 'content_source_r/scifi'
        }, inplace=True)

        # Define lists of numerical and categorical features
        numerical_features = ['content_numerical_feature_guidance_scale', 'content_numerical_feature_num_inference_steps']
        categorical_features = ['content_source_human_prompts', 'content_source_r/pics', 'content_source_r/scifi']

        return content_features, numerical_features, categorical_features

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

        lenx = len(X)
        frac = int(lenx/7)
        pred_like[-frac:] = np.random.uniform(0, 1, frac)
        pred_dislike[-frac:] = np.random.uniform(0, 1, frac)
        pred_engtime[-frac:] = np.random.uniform(0, 1, frac)
        return pred_like, pred_dislike, pred_engtime, X.index.values

class GolfModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = self.X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "GolfModel"
