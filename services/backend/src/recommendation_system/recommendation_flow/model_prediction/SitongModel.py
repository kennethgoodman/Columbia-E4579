import random
# from src import db
import pandas as pd 
import numpy as np
from collections import defaultdict
# from sqlalchemy.sql import text
import os
import pickle
import tensorflow_decision_forests as tfdf


class Model():

    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path + 'gbdt_model_v3.pickle', 'rb') as f:
            self.GBDT_model = pickle.load(f)
        with open(self.data_path + 'content_artist_style_dic.pickle', 'rb') as f:
            self.dic_id_style = pickle.load(f)
        with open(self.data_path + 'prediction_prep_dic.pickle', 'rb') as f:
            self.prep_dic = pickle.load(f) 
        with open(self.data_path + 'dic_id_to_embedding.pickle', 'rb') as f:
            self.dic_id_embed = pickle.load(f)


    def preprocess(self):
        # no need to run before every prediction
         
        engage['like'] = engage.apply(lambda row: pd.Series(1 if row['engagement_type'] == 'Like' and
                                                      row['engagement_value'] == 1 else 0), axis=1)
        engage['dislike'] = engage.apply(lambda row: pd.Series(1 if row['engagement_type'] == 'Like' and
                                                         row['engagement_value'] == -1 else 0), axis=1)
        engage['content_total_likes'] = engage.groupby('content_id')['like'].transform('sum')

        engage['content_total_dislikes'] = engage.groupby('content_id')['dislike'].transform('sum')
        engage['user_total_likes'] = engage.groupby('user_id')['like'].transform('sum')
        
        dic = defaultdict(dict)
        dic['content_total_likes'] = dict(zip(engage.content_id, engage.content_total_likes))
        dic['content_total_dislikes'] = dict(zip(engage.content_id, engage.content_total_dislikes))
        dic['user_total_likes'] = dict(zip(engage.content_id, engage.user_total_likes))

        with open(data_path + 'prediction_prep_dic.pickle', 'wb') as f:
            pickle.dump(dic, f)


    def predict_engage(self, content_ids, user_id, score=None, **kwargs):

        df = pd.DataFrame(content_ids,columns=['content_id'])
        df['user_id'] = str(user_id)  
        df['style'] = df['content_id'].apply(lambda x: self.dic_id_style[x])

        df['content_total_likes'] = df['content_id'].apply(lambda x: self.prep_dic['content_total_likes'][x])
        content_total_likes_dic = dict(zip(df['content_id'], df['content_total_likes']))

        df['content_total_dislikes'] = df['content_id'].apply(lambda x: self.prep_dic['content_total_dislikes'][x])
        content_total_dislikes_dic = dict(zip(df['content_id'], df['content_total_dislikes']))

        df['user_total_likes'] = df['content_id'].apply(lambda x: self.prep_dic['user_total_likes'][x])

        df['explore'] = (df['content_total_likes'] + df['content_total_dislikes'])/2 -abs(df['content_total_likes'] - df['content_total_dislikes'])
        explore_dic = dict(zip(df['content_id'], df['explore']))
        df = df.drop('explore',axis=1)

        # attach embed matrix
        train_content_ids = df.content_id.tolist()
        embed_matrix = []
        for content_id in train_content_ids:
            embed_matrix.append(self.dic_id_embed[content_id])
        embed_matrix = pd.DataFrame(embed_matrix)

        df = pd.concat([df, embed_matrix], axis=1)
        df = df.drop(['content_id'], axis=1)
        df['user_id'] = df['user_id'].astype(str)
        df['style'] = df['style'].astype(str)
        df.columns = [str(c) for c in df.columns]
        print("df.shape:", df.shape)
        
        tf_data = tfdf.keras.pd_dataframe_to_tf_dataset(df,task=tfdf.keras.Task.REGRESSION)
        pred_series = self.GBDT_model.predict(tf_data)

        content_to_engage_dic = dict(zip(content_ids, pred_series))
        return list(
            map(
                lambda content_id: {
                    "content_id": content_id,
                    # probability 0~1
                    "p_engage": float(content_to_engage_dic[content_id]),
                    "style": self.dic_id_style[content_id],
                    "total_likes": content_total_likes_dic[content_id],
                    "total_dislikes": content_total_dislikes_dic[content_id],
                    "explore": explore_dic[content_id],
                    "score": kwargs.get("scores", {}).get(content_id, {}).get("score", None)
                },
                content_ids,
            ) 
        )


# if __name__ == "__main__":
#     base_path = os.getcwd() 
#     data_path = base_path + '/data/local_data/'

#     m = Model(data_path)

#     ## m.preprocess() # don't need to run every time

#     ## for testing
#     engage = pd.read_csv(data_path + 'engagement.csv')[['user_id','content_id','engagement_type','engagement_value']]
#     test_content_ids = engage['content_id'].tolist()[:100]
#     test_user_id = 10 

#     print("Prediction started....")
#     res = m.predict_engage(test_content_ids, test_user_id)
#     print("Prediction finished....")
    
#     print(res)
#     with open(data_path + 'prediction_output_2.pickle', 'wb') as f:
#         pickle.dump(res, f)
    


