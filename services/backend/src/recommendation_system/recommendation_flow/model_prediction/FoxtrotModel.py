import numpy as np
import pandas as pd
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from src.recommendation_system.ml_models.foxtrot_lgb_model.Foxtrotlgb_model import (
    ModelController,
)
from functools import lru_cache

from .AbstractModel import AbstractModel

# load model once
model = ModelController("lgb", load_model=True).model

# load data once
CONTENT_FEATURES = pd.DataFrame()
USER_FEATURES = pd.DataFrame()


@lru_cache(1)
def read_feature():
    global CONTENT_FEATURES, USER_FEATURES
    if os.path.isfile("Foxtrot_content_features.pkl"):
        print("reading data from Foxtrot_content_features.pkl")
        with open("Foxtrot_content_features.pkl", "rb") as f:
            CONTENT_FEATURES = pickle.load(f)
    if os.path.isfile("Foxtrot_user_features.pkl"):
        print("reading data from Foxtrot_user_features.pkl")
        with open("Foxtrot_user_features.pkl", "rb") as f:
            USER_FEATURES = pickle.load(f)


class FoxtrotModel(AbstractModel):
    def __init__(self):
        super(FoxtrotModel, self).__init__
        self.content_features = CONTENT_FEATURES
        self.user_features = USER_FEATURES

    def _create_idv_data(self, content_id, user_id):
        # return np.array([content_id, user_id])
        user_feature = self.user_features.loc[user_id]
        item_feature = self.content_features.loc[content_id]

        user_embedding = np.array(user_feature["embedding"])
        user_embedding_feature = user_embedding[[5, 25, 66]]

        item_embedding = np.array(item_feature["embedding"])
        item_embedding_feature = user_embedding[[0, 11]]

        similarity = cosine_similarity(
            user_embedding.reshape(-1, 512), item_embedding.reshape(-1, 512)
        )[0]

        feature = np.hstack(
            [
                similarity,
                user_feature.values[:-1],
                item_feature.values[:-1],
                user_embedding_feature,
                item_embedding_feature,
            ]
        )

        return feature

    def _create_all_data(self, content_ids, user_id):
        return np.array(
            list(
                map(
                    lambda content_id: self._create_idv_data(content_id, user_id),
                    content_ids,
                )
            )
        ).reshape((len(content_ids), 12))

    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        return list(
            map(
                lambda i: {
                    "content_id": content_ids[i],
                    "p_engage": random.random(),  # hard coding that first output is p(Engage | data)
                    "score": kwargs.get("score", None),
                },
                range(len(content_ids)),
            )
        )
