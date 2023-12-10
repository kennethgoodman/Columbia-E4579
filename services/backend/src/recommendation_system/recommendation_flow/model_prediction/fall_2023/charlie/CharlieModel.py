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

with open(legalize('charlie_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)


with open(legalize('charlie_postprocessor.pkl'), 'rb') as f:
    POST_PROCESSOR = pickle.load(f)

class CharlieFeatureGeneration(AbstractFeatureGeneration):
    def feature_generation_user(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Generates user features. Keep all the categorical variables as is, since the one-hot encoding will be done
        by our own pipeline. Along with the feature dataframe, you'll need to output lists of numerical features
        and categorical features as well.
        Returns:
            pd.DataFrame: User feature dataframe
            List[str]: List of numerical features. E.g. ['feat_1', 'feat_3', ...]
            List[str]: List of categorical features. E.g. ['feat_2', 'feat_4', ...]
        """
        user_data_copy = self.user_data.drop_duplicates().copy()
        e_time = user_data_copy[user_data_copy["engagement_type"] == "MillisecondsEngagedWith"]
        e_time.reset_index(drop=True, inplace=True)
        content_count = pd.DataFrame(e_time[["user_id", "content_id"]].groupby("user_id").size())
        e_time_per_user = e_time[["user_id", "engagement_value"]].groupby("user_id").mean()
        like_dislike = user_data_copy[user_data_copy["engagement_type"] != "MillisecondsEngagedWith"]
        like_dislike.rename(columns={"engagement_value": "reaction"}, inplace=True)
        like_dislike = like_dislike.merge(e_time[["content_id", "user_id", "engagement_value"]], on=["content_id", "user_id"])
        like_dislike.drop(columns=["engagement_metadata", "created_date"], inplace=True)
        like = like_dislike[like_dislike.reaction == 1]
        dislike = like_dislike[like_dislike.reaction == -1]
        like_avg_time_per_user = like[["user_id", "engagement_value"]].groupby("user_id").mean()
        dislike_avg_time_per_user = dislike[["user_id", "engagement_value"]].groupby("user_id").mean()
        feature_df = content_count.merge(e_time_per_user, left_index=True, right_index=True, how="outer")
        feature_df = feature_df.merge(like_avg_time_per_user, left_index=True, right_index=True, how="left")
        feature_df = feature_df.merge(dislike_avg_time_per_user, left_index=True, right_index=True, how="left")
        feature_df.columns = ["content_count", "avg_engagement_time", "like_avg_engagement_time", "dislike_avg_engagement_time"]
        feature_df.reset_index(drop=False, inplace=True)
        feature_df = feature_df.fillna(0)
        return feature_df, ["content_count", "avg_engagement_time", "like_avg_engagement_time", "dislike_avg_engagement_time"], []

    def feature_generation_content(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Generates content features. Keep all the categorical variables as is, since the one-hot encoding will be
        done by our own pipeline. Along with the feature dataframe, you'll need to output lists of numberical
        features and categorical features as well.
        Returns:
            pd.DataFrame: User feature dataframe
            List[str]: List of numerical features. E.g. ['feat_1', 'feat_3', ...]
            List[str]: List of categorical features. E.g. ['feat_2', 'feat_4', ...]
        """
        feature_df = self.generated_content_metadata.drop_duplicates().copy()
        top_artist_styles = feature_df["artist_style"].value_counts().nlargest(50)
        top_sources = feature_df["source"].value_counts().nlargest(50)
        feature_df["artist_style"] = feature_df["artist_style"].apply(
            lambda x: x if x in top_artist_styles else "other_artist_style"
        )
        feature_df["source"] = feature_df["source"].apply(
            lambda x: x if x in top_sources else "other_source"
        )
        num_inference_steps_mapping = {20: "Low", 50: "Medium", 75: "Relatively_high", 100: "High"}
        feature_df["num_inference_steps_level"] = feature_df["num_inference_steps"].apply(lambda x: num_inference_steps_mapping[x])
        feature_df.drop("num_inference_steps", axis=1, inplace=True)
        return feature_df, ["guidance_scale"], ["num_inference_steps_level", "artist_style", "source"]

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
        pred_like_proba = model['like'].predict_proba(X)
        pred_like = [p[1] for p in pred_like_proba]
        pred_dislike_proba = model['dislike'].predict_proba(X)
        pred_dislike = [p[1] for p in pred_dislike_proba]
        pred_engtime = model['engage_time'].predict(X)
        return pred_like, pred_dislike, pred_engtime, X.index.values


class CharlieModel(AbstractModel):
    def _predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        X = kwargs['fg'].X.loc[user_id].loc[content_ids]
        return kwargs['fg'].predict_probabilities(X)

    def _get_name(self):
        return "CharlieModel"
