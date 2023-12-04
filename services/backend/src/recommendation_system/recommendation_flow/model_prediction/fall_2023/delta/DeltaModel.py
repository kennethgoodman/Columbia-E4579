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


with open(legalize('delta_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

with open(legalize('delta_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)


class DeltaFeatureGeneration(AbstractFeatureGeneration):
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

        # Filtering like/dislike engagements
        like_data = self.user_data[self.engagement_data['engagement_type'] == 'Like']

        # Grouping by 'user_id' and 'content_id' and getting the latest engagement for each pair
        latest_like_data = like_data.sort_values('created_date').groupby(['user_id', 'content_id']).tail(1)

        # Getting total likes for each user
        like_engagements = latest_like_data[(latest_like_data['engagement_value']==1)].copy()
        like_feature_df = like_engagements.groupby('user_id')['engagement_value'].sum().reset_index()
        like_feature_df.rename(columns={'engagement_value': 'user_likes'}, inplace=True)
        # Fill NaN values with 0 (users with no "like" engagements)
        like_feature_df['user_likes'].fillna(0, inplace=True)


        # Getting total dislikes for each user
        dislike_engagements = latest_like_data[(latest_like_data['engagement_value']==-1)].copy()
        dislike_feature_df = dislike_engagements.groupby('user_id')['engagement_value'].sum().reset_index()
        dislike_feature_df.rename(columns={'engagement_value': 'user_dislikes'}, inplace=True)
        # Fill NaN values with 0 (users with no "dislike" engagements)
        dislike_feature_df['user_dislikes'].fillna(0, inplace=True)

        # Getting average engage time for each user
        time_engagements = self.user_data[self.user_data['engagement_type'] == 'MillisecondsEngagedWith'].copy()

        # consider each user's max engagement time with each content
        time_engagements = time_engagements.groupby(['user_id', 'content_id'])['engagement_value'].max().reset_index()
        engage_feature_df = time_engagements.groupby('user_id')['engagement_value'].mean().reset_index()
        engage_feature_df.rename(columns={'engagement_value': 'user_engagetime'}, inplace=True)
        # fill NaN values with avg_engage (users with no engagment time data)
        avg_engage = engage_feature_df['user_engagetime'].mean()
        engage_feature_df['user_engagetime'].fillna(avg_engage, inplace=True)

        feature_df = pd.merge(like_feature_df, dislike_feature_df , on='user_id', how='left')
        feature_df = pd.merge(feature_df, engage_feature_df , on='user_id', how='left')

        return feature_df, ['user_likes', 'user_dislikes', 'user_engagetime'], []


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

        feature_df = self.generated_content_metadata.copy()

        # numerical feature 1: (average) guidance scale
        mean_engage = feature_df["guidance_scale"].mean()
        guide_df = feature_df.groupby('content_id')['guidance_scale'].mean().reset_index()
        guide_df = guide_df.rename(columns={'guidance_scale': 'content_guidance_scale'})
        feature_df = pd.merge(feature_df, guide_df, on='content_id', how='left')
        feature_df['content_guidance_scale'].fillna(mean_engage, inplace=True)

        # numerical feature 2: num inference steps
        mean_inf = feature_df["num_inference_steps"].mean()
        inf_df = feature_df.groupby('content_id')['num_inference_steps'].mean().reset_index()
        inf_df = inf_df.rename(columns={'num_inference_steps': 'content_inference_steps'})
        feature_df = pd.merge(feature_df, inf_df, on='content_id', how='left')
        feature_df['content_inference_steps'].fillna(mean_inf, inplace=True)


        # categorical feature 1: source
        feature_df['content_source'] = 'other'
        feature_df.loc[feature_df['source'] == 'human_prompts', 'content_source'] = 'human_prompts'
        feature_df.loc[feature_df['source'] == 'r/Showerthoughts', 'content_source'] = 'r/Showerthoughts'


        # categorical feature 2: artist style
        style_list = [
            'studio',
            'medieval',
            'anime',
            'kerry_james_marshall',
            'gta_v',
            'scifi',
            'van_gogh',
            'salvador_dali',
            'jean-michel_basquiat',
            'face_and_lighting'
        ]
        #style_list = ['movie', 'empty']
        feature_df['content_style'] = feature_df['artist_style']
        feature_df['content_style'].fillna("empty", inplace=True)
        feature_df.loc[feature_df['content_style'].str.startswith('movie:'), 'content_style'] = 'movie'
        feature_df.loc[~feature_df['content_style'].isin(style_list), 'content_style'] = 'other'


        return feature_df, ['content_inference_steps'], ['content_source', 'content_style']

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
        pred_like = MODEL['like'].predict(X).flatten()
        pred_dislike = MODEL['dislike'].predict(X).flatten()
        pred_engtime = MODEL['engage_time'].predict(X).flatten()
        return pred_like, pred_dislike, pred_engtime, X.index.values


class DeltaModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = kwargs['fg'].X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "DeltaModel"
