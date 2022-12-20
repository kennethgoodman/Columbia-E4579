import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import joblib


class EchoModel:

    def __init__(self):

        model_path = "src/echo_space/models/lgbm.pkl"
        self.model = joblib.load(model_path)
        user_path = "src/echo_space/processed_data/User_Features.parquet"
        content_path = "src/echo_space/processed_data/Content_Features.parquet"
        user_features = pd.read_parquet(user_path)
        content_features = pd.read_parquet(content_path)
        self.user_features = pd.concat(
            [user_features, pd.DataFrame(user_features["embed_combined"].tolist()).add_prefix('emb_')], axis=1)
        self.content_features = pd.concat(
            [content_features, pd.DataFrame(content_features["prompt_embedding"].tolist()).add_prefix('emb_')], axis=1)


    def _create_idv_data(self, content_id, user_id):
        '''
        return user features, content features and their consine similarities
        '''
        slct_user_feature = self.user_features[self.user_features["user_id"]== user_id]
        slct_content_feature = self.content_features[self.content_features["content_id"]== content_id]
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
        ).reshape((len(content_ids), -1))

    def predict_probabilities(self, user_id, content_ids, seed=None, **kwargs):
        predictions = self.model.predict_proba(self._create_all_data(content_ids, user_id))

        return list(zip(content_ids, predictions[:, 1]))

# user_id = 1
# content_ids = [28598, 28599]
# model_prediction = EchoModel()
# prob = model_prediction.predict_probabilities(user_id, content_ids)
# print(prob)

