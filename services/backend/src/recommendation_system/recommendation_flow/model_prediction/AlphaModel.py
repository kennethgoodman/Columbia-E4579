import random
import pandas as pd 
import numpy as np
from collections import defaultdict
import os
import pickle
import sys
import subprocess
from .AbstractModel import AbstractModel
import joblib
from sklearn.ensemble import RandomForestRegressor

def try_load_model(fn):
    try:
        with open(fn, 'rb') as f:
            return pickle.load(f)
    except:
        pass

dic_id_style = try_load_model('/usr/src/app/src/alpha/content_artist_style_dic.pickle')

try:
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        "tensorflow",
        "keras",
        "tensorflow_decision_forests"
    ])
    # GBDT_model = try_load_model('/usr/src/app/src/alpha/gbdt_model_v3.pickle')
    prep_dic = try_load_model('/usr/src/app/src/alpha/prediction_prep_dic.pickle')
    dic_id_embed = try_load_model("/usr/src/app/id_to_embedding.pkl")
    # load the random forest model
    loaded_rf = joblib.load("/usr/src/app/src/alpha/rfmodel.joblib")
except:
    pass


class AlphaModel(AbstractModel):
    def predict_probabilities(self, content_ids, user_id, seed=None, **kwargs):
        try:
            import tensorflow_decision_forests as tfdf
            import joblib
            from sklearn.ensemble import RandomForestRegressor
            df = pd.DataFrame(content_ids,columns=['content_id'])
            df['user_id'] = user_id
            df['content_total_likes'] = df['content_id'].apply(lambda x: prep_dic['content_total_likes'][x])
            content_total_likes_dic = dict(zip(df['content_id'], df['content_total_likes']))

            df['content_total_dislikes'] = df['content_id'].apply(lambda x: prep_dic['content_total_dislikes'][x])
            content_total_dislikes_dic = dict(zip(df['content_id'], df['content_total_dislikes']))

            df['user_total_likes'] = df['content_id'].apply(lambda x: prep_dic['user_total_likes'][x])
            df['explore'] = (df['content_total_likes'] + df['content_total_dislikes'])/2 -abs(df['content_total_likes'] - df['content_total_dislikes'])
            explore_dic = dict(zip(df['content_id'], df['explore']))
            df = df.drop('explore',axis=1)

            # attach embed matrix
            train_content_ids = df.content_id.tolist()
            dic_id_to_embeddings = {v[0]:v[1] for v in dic_id_embed}
            embed_matrix = []
            for content_id in train_content_ids:
                embed_matrix.append(dic_id_to_embeddings[content_id])

            embed_matrix = pd.DataFrame(embed_matrix)

            df = pd.concat([df, embed_matrix], axis=1)
            df = df.drop(['content_id'], axis=1)
            df.columns = [str(c) for c in df.columns]
            pred_series = list(loaded_rf.predict(df))
            content_to_engage_dic = dict(zip(content_ids, pred_series))
            return_val = list(
                map(
                    lambda content_id: {
                        "content_id": content_id,
                        # probability 0~1
                        # added some variability
                        "p_engage": float(content_to_engage_dic[content_id] + random.uniform(-0.25, 0.25)),
                        "style": dic_id_style[content_id],
                        "total_likes": content_total_likes_dic[content_id],
                        "total_dislikes": content_total_dislikes_dic[content_id],
                        "explore": explore_dic[content_id],
                        "score": kwargs.get("scores", {}).get(content_id, {}).get("score", None)
                    },
                    content_ids,
                )
            )
        except Exception as e:
            if seed:
                random.seed(seed)
            try: # in dev dic_id_style[content_id] has key error
                return_val = list(
                    map(
                        lambda content_id: {
                            "content_id": content_id,
                            "p_engage": random.random(),
                            "style": dic_id_style[content_id],
                            "score": kwargs.get("scores", {})
                            .get(content_id, {})
                            .get("score", None),
                        },
                        content_ids,
                    )
                )
            except KeyError:
                return_val = list(
                    map(
                        lambda content_id: {
                            "content_id": content_id,
                            "p_engage": random.random(),
                            "score": kwargs.get("scores", {})
                            .get(content_id, {})
                            .get("score", None),
                        },
                        content_ids,
                    )
                )
        return return_val
