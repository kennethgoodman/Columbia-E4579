import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, Normalizer
from typing import List
from typing import Tuple
import sys

class Postprocessor:
    def __init__(self,
                 numberical_features: List[str],
                 categorical_features: List[str]):

        self.numberical_features = numberical_features
        self.categorical_features = categorical_features

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encode_cols = []

    def fit(self, features_df: pd.DataFrame):

        self.scaler.fit(features_df[self.numberical_features])

        if len(self.categorical_features) > 0:
            self.encoder.fit(features_df[self.categorical_features])
            self.encode_cols = list(self.encoder.get_feature_names_out())

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:

        features_df[self.numberical_features] = self.scaler.transform(features_df[self.numberical_features])

        if len(self.categorical_features) > 0:
            features_df[self.encode_cols] = self.encoder.transform(features_df[self.categorical_features])

        return features_df

    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:

        self.fit(features_df)
        features_df = self.transform(features_df)

        return features_df
sys.modules['__main__'].Postprocessor = Postprocessor

class AbstractFeatureGeneration:
    def __init__(self, dc, content_ids):
        self.dc = dc
        self.content_ids = content_ids
        self.engagement_data, self.generated_content_metadata_data, self.user_data = (
            self.dc.return_data_copy()
        )
        self.engagement_data = self.engagement_data[
            self.engagement_data['content_id'].isin(content_ids)
        ]
        self.generated_content_metadata = self.generated_content_metadata_data = self.generated_content_metadata_data[
            self.generated_content_metadata_data['content_id'].isin(content_ids)
        ]
        self.postprocessor = self.load_postprocessor()
        self.set_X()

    def feature_generation_user(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        raise NotImplementedError("you need to implement this")

    def feature_generation_content(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        raise NotImplementedError("you need to implement this")

    def feature_eng(self):
        user_feature_df, user_num_feats, user_cat_feats = self.feature_generation_user()
        content_feature_df, content_num_feats, content_cat_feats = self.feature_generation_content()
        self.user_feature_df = user_feature_df
        self.content_feature_df = content_feature_df

        self.numerical_features = user_num_feats + content_num_feats
        self.categorical_features = user_cat_feats + content_cat_feats

        all_users = self.engagement_data['user_id'].drop_duplicates().tolist()
        all_contents = self.generated_content_metadata_data['content_id'].drop_duplicates().tolist()

        interaction_pairs = [(u, c) for u in all_users for c in all_contents]
        interaction_pairs = pd.DataFrame(interaction_pairs, columns=['user_id', 'content_id'])

        features_df = pd.merge(interaction_pairs,
                               user_feature_df, on='user_id', how='left')

        features_df = pd.merge(features_df,
                               content_feature_df, on='content_id', how='left')
        return features_df

    def load_postprocessor(self):
        raise NotImplementedError("you need to implement")

    def postprocess_feature(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Applied postprocessings (one-hot encoding & scaler) to the feature dataframe.

        Args:
            features_df (pd.DataFrame): Input feature dataframe.
            is_train (bool): Whether in training mode. If True, will fit the
                Postprocessor() and save to a pickle file. Else, will load the
                saved Postprocessor() and use it.

        Returns:
            pd.DataFrame: Output feature dataframe.
        """
        features_df = self.postprocessor.transform(features_df)

        self.all_numeric_features = self.numerical_features + self.postprocessor.encode_cols

        return features_df

    def set_X(self):
        features_df = self.feature_eng()
        for col in self.postprocessor.numberical_features + self.postprocessor.categorical_features:
            if col not in features_df:
                features_df[col] = 0.0
        features_df = self.postprocess_feature(features_df)
        self.X_all = features_df
        self.X = features_df.set_index(['user_id', 'content_id'])[self.all_numeric_features].fillna(0)

