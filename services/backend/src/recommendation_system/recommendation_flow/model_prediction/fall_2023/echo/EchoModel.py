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


import sys
import torch.nn as nn
class RankingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RankingModel, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        prediction = self.output_layer(x.float())

        return prediction
sys.modules['__main__'].RankingModel = RankingModel
with open(legalize('echo_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

with open(legalize('echo_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)

TOP_ARTIST_STYLES = 30
TOP_SOURCES = 30
TOP_CONTENT = 100

class EchoFeatureGeneration(AbstractFeatureGeneration):
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

        df = self.user_data.drop_duplicates()
        top_n_content = [81576, 112990, 115079, 29667, 116285, 119163, 121340, 98660, 112723, 112718, 100600, 123413,
                         84509, 124241, 85954, 91035, 96308, 107234, 124197, 129220, 97383, 32858, 101707, 83115, 82523,
                         91267, 112943, 115042, 85369, 86154, 96136, 113087, 125842, 118824, 115641, 124704, 87170,
                         124089, 55132, 95154, 92391, 94627, 95448, 96061, 118478, 95420, 90800, 85672, 76810, 79978,
                         73723, 85261, 112433, 85495, 102788, 121219, 124002, 37896, 31280, 89381, 118564, 132295,
                         31278, 101773, 130301, 111201, 97499, 97627, 108116, 87998, 62821, 86786, 86593, 118721, 87250,
                         33369, 91588, 95370, 121439, 124954, 85949, 123170, 124374, 129087, 58001, 79847, 90687,
                         101576, 90213, 104604, 106588, 115754, 117136, 117157, 127128, 104392, 86289, 91755, 116519,
                         85093]
        from collections import defaultdict
        user_vector_dict = defaultdict(lambda: {
            'millisecond_engaged_vector': np.zeros(len(top_n_content)),
            'like_vector': np.zeros(len(top_n_content)),
            'dislike_vector': np.zeros(len(top_n_content))
        })

        def aggregate_engagement(group):
            millisecond_engagement_sum = group.loc[group['engagement_type'] != 'Like', 'engagement_value'].sum()
            likes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == 1)].shape[0]
            dislikes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == -1)].shape[
                0]

            return pd.Series({
                'millisecond_engagement_sum': millisecond_engagement_sum,
                'likes_count': likes_count,
                'dislikes_count': dislikes_count
            })

        for user_id in df['user_id'].unique():
            user_vector_dict[user_id]['millisecond_engaged_vector'] = np.zeros(len(top_n_content))
            user_vector_dict[user_id]['like_vector'] = np.zeros(len(top_n_content))
            user_vector_dict[user_id]['dislike_vector'] = np.zeros(len(top_n_content))
        if len(df[df['content_id'].isin(top_n_content)]) != 0:
            engagement_aggregate = df[df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id']).apply(
                aggregate_engagement).reset_index()
            for _, row in engagement_aggregate.iterrows():
                user_id = row['user_id']
                content_id = row['content_id']
                idx = top_n_content.index(content_id)

                user_vector_dict[user_id]['millisecond_engaged_vector'][idx] = row['millisecond_engagement_sum']
                user_vector_dict[user_id]['like_vector'][idx] = row['likes_count']
                user_vector_dict[user_id]['dislike_vector'][idx] = row['dislikes_count']

        user_vector_df = pd.DataFrame.from_dict(user_vector_dict, orient='index')
        millisecond_columns = [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
        like_columns = [f"like_vector_{i}" for i in range(TOP_CONTENT)]
        dislike_columns = [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]

        user_vector_df1 = pd.DataFrame(user_vector_df['millisecond_engaged_vector'].tolist(),
                                       index=user_vector_df.index, columns=millisecond_columns)
        user_vector_df2 = pd.DataFrame(user_vector_df['like_vector'].tolist(), index=user_vector_df.index,
                                       columns=like_columns)
        user_vector_df3 = pd.DataFrame(user_vector_df['dislike_vector'].tolist(), index=user_vector_df.index,
                                       columns=dislike_columns)

        feature_df = pd.concat([user_vector_df1, user_vector_df2, user_vector_df3], axis=1)
        feature_df = feature_df.reset_index().rename(columns={'index': 'user_id'})

        return feature_df, millisecond_columns + like_columns + dislike_columns, []

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

        feature_df = self.generated_content_metadata.drop_duplicates(subset='content_id', keep='first')

        top_artist_styles = ['medieval', 'anime', 'studio', 'shepard_fairey', 'oil_on_canvas', 'unreal_engine',
                             'edward_hopper', 'keith_haring', 'detailed_portrait', 'van_gogh', 'jackson_pollock',
                             'juan_gris', 'scifi', 'ibrahim_el_salahi', 'ma_jir_bo', 'franck_slama',
                             'jean-michel_basquiat', 'kerry_james_marshall', 'marta_minujín', 'face_and_lighting',
                             'salvador_dali', 'leonardo_da_vinci', 'takashi_murakami', 'gta_v', 'laura_wheeler_waring',
                             'louise bourgeois', 'movie: Dances-with-Wolves', 'movie: Interstellar',
                             'movie: Indiana-Jones-IV', 'movie: Gravity']
        top_sources = ['human_prompts', 'r/EarthPorn', 'r/Showerthoughts', 'r/scifi', 'r/pics',
                       'r/Damnthatsinteresting', 'r/educationalgifs', 'r/MadeMeSmile', 'r/SimplePrompts',
                       'r/RetroFuturism', 'r/AccidentalArt', 'r/Cyberpunk', 'Dances-with-Wolves', 'Buddha',
                       'Interstellar', 'Napoleon Hill', 'Abraham Lincoln', 'r/oddlysatisfying', 'Indiana-Jones-IV',
                       'Gravity', 'Winston Churchill', 'Ralph Waldo Emerson', 'r/whoahdude', 'Confucius', 'Batman',
                       'Johann Wolfgang von Goethe', 'Tombstone', 'Godfather', 'Legend-of-Darkness', 'Ni-vu-ni-connu']

        feature_df['artist_style'] = feature_df['artist_style'].apply(
            lambda x: x if x in top_artist_styles else 'other')
        feature_df['source'] = feature_df['source'].apply(lambda x: x if x in top_sources else 'other')

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

        model = MODEL
        import torch
        def sigmoid(x):
            return torch.exp(x) / (1 + torch.exp(x))

        model.eval()

        with torch.no_grad():
            preds = model(torch.Tensor(X.to_numpy()))

        return (
            sigmoid(preds[:, 0]).flatten().numpy(),
            sigmoid(preds[:, 1]).flatten().numpy(),
            torch.exp(preds[:, 2]).flatten().numpy(),
            X.index.values
        )

class EchoModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = self.X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "EchoModel"
