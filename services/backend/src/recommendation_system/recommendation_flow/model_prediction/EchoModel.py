import numpy as np
import pandas as pd
import pickle
from .AbstractModel import AbstractModel
from numpy import dot
from numpy.linalg import norm

# load model and features once
model_path = "./services/backend/src/recommendation_system/ml_models/Lightgbm_model_1219.sav"
model = pickle.load(open(model_path, 'rb'))
user_path = "./services/backend/processed_data/Content_Features.parquet"
content_path = "./services/backend/processed_data/Content_Features.parquet"
user_features = pd.read_parquet(user_path)
content_features = pd.read_parquet(content_path)

class EchoModel(AbstractModel):
    def _create_idv_data(self, content_id, user_id, content_features, user_features):
        '''
        return user features, content features and their consine similarities
        '''
        slct_user_feature = user_features[user_features["user_id"]== user_id]
        slct_content_feature = content_features[content_features["content_id"]== content_id]
        user_embedding_lst = slct_user_feature["embed_combined"].tolist()[0]
        content_feature_lst = slct_content_feature["prompt_embedding"].tolist()[0]
        a,b = user_embedding_lst, content_feature_lst
        # Calculate the similarity of user embedding and content_embedding
        cosine_similarity = dot(a, b)/(norm(a)*norm(b))
        # full features include user features, content features
        full_features = slct_user_feature.drop(["user_id","embed_combined"],axis=1).values.tolist()[0]\
                        + slct_content_feature.drop(["content_id","source","prompt_embedding"],axis=1).values.tolist()[0] \
                        + [cosine_similarity]
        return full_features
        


    def _create_all_data(self, content_ids, user_id):
        return np.array(
            list(
                map(
                    lambda content_id: self._create_idv_data(content_id, user_id),
                    content_ids,
                )
            )
        ).reshape((len(content_ids), 1037))

    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        predictions = model.predict_proba(self._create_all_data(content_ids, user_id))
        return list(
            map(
                lambda i: {
                    "content_id": content_ids[i],
                    "p_engage": predictions[i][
                        1
                    ],  # hard coding that first output is p(Engage | data)
                    "score": kwargs.get("score", None),
                },
                range(len(content_ids)),
            )
        )   
