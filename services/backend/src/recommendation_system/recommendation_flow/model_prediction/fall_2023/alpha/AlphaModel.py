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

with open(legalize('alpha_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)


with open(legalize('alpha_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)

class AlphaFeatureGeneration(AbstractFeatureGeneration):
    def get_Ys(self, engagement_data) -> pd.DataFrame:
        """Engineers taget variable that you are predicting.
        Args
            engagement_data (pd.DataFrame): Engagement data.
        Returns
            pd.DataFrame: Dataframe of 5 columns;
                'user_id', 'content_id', 'like', 'dislike', 'engage_time'
        """

        np.random.seed(42)

        target_df = engagement_data[['user_id', 'content_id']].drop_duplicates().copy()

        def map_like_dislike(row):
            if str(row['engagement_type']) == 'Like':
                if row['engagement_value'] > 0:
                    return 1
                elif row['engagement_value'] < 0:
                    return -1
            else:
                return None

        def map_engagement_time(row):
            if str(row['engagement_type']) == 'MillisecondsEngagedWith':
                return row['engagement_value']
            else:
                return None

        engagement_data["engagement_like_dislike"] = engagement_data.apply(map_like_dislike, axis=1)
        engagement_data["engage_time"] = engagement_data.apply(map_engagement_time, axis=1)
        like_dislike = engagement_data[engagement_data['engagement_type'] == 'Like'][
            ['user_id', 'content_id', 'engagement_like_dislike']].groupby(['user_id', 'content_id']).max().reset_index()
        engagement_time = engagement_data[engagement_data['engagement_type'] == 'MillisecondsEngagedWith'][
            ['user_id', 'content_id', 'engage_time']].groupby(['user_id', 'content_id']).sum().reset_index()

        target_df = target_df.merge(engagement_time, on=('user_id', 'content_id'), how='left').merge(like_dislike, on=(
        'user_id', 'content_id'), how='left').fillna(0)

        target_df['like'] = (target_df['engagement_like_dislike'] > 0).astype(int)
        target_df['dislike'] = (target_df['engagement_like_dislike'] < 0).astype(int)
        target_df = target_df.drop('engagement_like_dislike', axis=1)

        return target_df

    def feature_generation_user(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        ## TODO: Fix NaN Values
        """Generates user features. Keep all the categorical variables as is,
        since the one-hot encoding will be done by our own pipeline. Along with
        the feature dataframe, you'll need to output lists of numberical features
        and categorical features as well.

        Returns
          pd.DataFrame: User feature dataframe
          List[str]: List of numerical features. E.g. ['feat_1', 'feat_3, ...]
          List[str]: List of categorical features. E.g. ['feat_2', 'feat_4, ...]
        """
        target_df = self.get_Ys(self.user_data)

        user_df = self.user_data
        user_avg_engagement_time = target_df[['user_id', 'engage_time']].groupby(['user_id']).mean().rename(
            columns={'engage_time': 'user_avg_engagement_time'})
        user_like_rate = target_df[['user_id', 'like']].groupby(['user_id']).mean().rename(
            columns={'like': 'user_like_rate'})
        user_dislike_rate = target_df[['user_id', 'dislike']].groupby(['user_id']).mean().rename(
            columns={'dislike': 'user_dislike_rate'})
        frequency = user_df.groupby(['user_id']).size().reset_index(name='frequency')
        content_diversity = user_df.groupby('user_id')['content_id'].nunique().reset_index(name='content_diversity')
        duration_variability = user_df[user_df['engagement_type'] == 'MillisecondsEngagedWith'].groupby('user_id')[
            'engagement_value'].std().reset_index(name='duration_variability').fillna(0)

        feature_df = self.user_data[['user_id']].drop_duplicates().copy()
        feature_df = feature_df.merge(user_avg_engagement_time, on='user_id')
        feature_df = feature_df.merge(user_like_rate, on='user_id')
        feature_df = feature_df.merge(user_dislike_rate, on='user_id')
        feature_df = feature_df.merge(frequency, on='user_id')
        feature_df = feature_df.merge(content_diversity, on='user_id')
        feature_df = feature_df.merge(duration_variability, on='user_id')

        return feature_df, ['user_avg_engagement_time', 'user_like_rate', 'user_dislike_rate', 'frequency',
                            'content_diversity', 'duration_variability'], []

    def feature_generation_content(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        # TODO add more features
        """Generates content features. Keep all the categorical variables as is,
        since the one-hot encoding will be done by our own pipeline. Along with
        the feature dataframe, you'll need to output lists of numberical features
        and categorical features as well.

        Returns
          pd.DataFrame: User feature dataframe
          List[str]: List of numerical features. E.g. ['feat_1', 'feat_3, ...]
          List[str]: List of categorical features. E.g. ['feat_2', 'feat_4, ...]
        """

        target_df = self.get_Ys(self.engagement_data)
        content_avg_engagement_time = target_df[['content_id', 'engage_time']].groupby(['content_id']).mean().rename(
            columns={'engage_time': 'content_avg_engagement_time'})
        content_like_rate = target_df[['content_id', 'like']].groupby(['content_id']).mean().rename(
            columns={'like': 'content_like_rate'})
        content_dislike_rate = target_df[['content_id', 'dislike']].groupby(['content_id']).mean().rename(
            columns={'dislike': 'content_dislike_rate'})

        # guidance_scale = content_df.groupby('content_id')['guidance_scale'].apply(lambda x: x.mode()[0])
        # num_inference_steps = content_df.groupby('content_id')['num_inference_steps'].apply(lambda x: x.mode()[0])
        source_list = ['human_prompts', 'r/Showerthoughts', 'r/EarthPorn', 'r/scifi', 'r/pics',
                       'r/Damnthatsinteresting', 'r/MadeMeSmile', 'r/educationalgifs', 'r/SimplePrompts']
        style_list = ['van_gogh', 'jean-michel_basquiat', 'detailed_portrait',
                      'kerry_james_marshall', 'medieval', 'studio', 'edward_hopper',
                      'takashi_murakami', 'anime', 'leonardo_da_vinci',
                      'laura_wheeler_waring', 'ma_jir_bo', 'jackson_pollock',
                      'shepard_fairey', 'unreal_engine', 'face_and_lighting', 'keith_haring',
                      'marta_minujín', 'franck_slama', 'oil_on_canvas', 'scifi', 'gta_v',
                      'louise bourgeois', 'salvador_dali', 'ibrahim_el_salahi', 'juan_gris']

        def map_source(row):
            if row['source'] in source_list:
                return row['source']
            else:
                return 'other'

        def map_style(row):
            if row['artist_style'] in source_list:
                return row['artist_style']
            else:
                return 'other'

        feature_df = self.generated_content_metadata[
            ['content_id', 'guidance_scale', 'num_inference_steps', 'source', 'artist_style']].drop_duplicates().copy()
        feature_df['source'] = feature_df.apply(map_source, axis=1)
        feature_df['artist_style'] = feature_df.apply(map_source, axis=1)
        feature_df = feature_df.merge(content_avg_engagement_time, on='content_id')
        feature_df = feature_df.merge(content_like_rate, on='content_id')
        feature_df = feature_df.merge(content_dislike_rate, on='content_id')

        return feature_df, ['content_avg_engagement_time', 'content_like_rate', 'content_dislike_rate',
                            'guidance_scale', 'num_inference_steps'], ['source', 'artist_style']

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
        pred_like = MODEL['like'].predict_proba(X)[:, 0]
        pred_dislike = MODEL['dislike'].predict_proba(X)[:, 0]
        pred_engtime = MODEL['engage_time'].predict(X)
        return pred_like, pred_dislike, pred_engtime, X.index.values


class AlphaModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = kwargs['fg'].X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "AlphaModel"
