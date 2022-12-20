from typing import List
from .AbstractRanker import AbstractRanker
import heapq
from sqlalchemy.engine import create_engine
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

engine = create_engine('mysql://mysql:mysql@127.0.0.1:3307/api_dev')
engagement = pd.read_sql('select * from api_dev.engagement', engine)
metadata = pd.read_sql('select * from api_dev.generated_content_metadata', engine)

class EchoRanker:
    def rank_ids(self, limit, probabilities, user_id, seed, starting_point) -> List[int]:
        uuid = user_id
        like = engagement.loc[(engagement['user_id']==uuid) 
                            & (engagement['engagement_type']=='Like') 
                            & (engagement['engagement_value']==1), 
                            'content_id'].tolist()
        dislike = engagement.loc[(engagement['user_id']==uuid) 
                            & (engagement['engagement_type']=='Like') 
                            & (engagement['engagement_value']==-1), 
                            'content_id'].tolist()
        not_interested = engagement.loc[(engagement['user_id']==uuid) 
                            & (engagement['engagement_type']=='MillisecondsEngagedWith'), 
                            'content_id'].tolist()
        # take the subset are mutually exclusive to the other categories
        not_interested = list(set(not_interested) - set(like) - set(dislike))
        
        # content_ids = {
        #     'like':like,
        #     'dislike':dislike,
        #     'not_interested':not_interested
        # }
        
        # like_embeddings = list(
        #     map(
        #         lambda content_id:
        #         json.loads(metadata.loc[metadata['content_id']==content_id,'prompt_embedding'].values[0])
        #         ,content_ids['like']
        #         )
        #     )
        # dislike_embeddings = list(
        #     map(
        #         lambda content_id:
        #         json.loads(metadata.loc[metadata['content_id']==content_id,'prompt_embedding'].values[0])
        #         ,content_ids['dislike']
        #         )
        #     )
        # not_interested_embeddings = list(
        #     map(
        #         lambda content_id:
        #         json.loads(metadata.loc[metadata['content_id']==content_id,'prompt_embedding'].values[0])
        #         ,content_ids['not_interested']
        #         )
        #     )
        # total_embeddings = np.vstack((
        #     like_embeddings,
        #     dislike_embeddings,
        #     not_interested_embeddings
        # ))
        
        # like_cos_sim = cosine_similarity(like_embeddings)
        # dislike_cos_sim = cosine_similarity(dislike_embeddings)
        # not_interested_cos_sim = cosine_similarity(not_interested_embeddings)
        # total_cos_sim = cosine_similarity(total_embeddings)

        # like2total_cos_sim_std = like_cos_sim.std() / total_cos_sim.std()
        # dislike2total_cos_sim_std = dislike_cos_sim.std() / total_cos_sim.std()
        # not_interested2total_cos_sim_std = not_interested_cos_sim.std() / total_cos_sim.std()

        # ------------------------- rank by popularity score --------------------
        all_content_ids = list(map(lambda x: x["content_id"], probabilities))
        
        content_id_to_score = {}
        
        for content_id in all_content_ids:
            occurrance = engagement.loc[
                engagement['content_id'] == content_id,
                :
            ]
            # assumption: 
            # avg time spent per view, the more time spent per view, the more popular the image is
            content_id_to_score[content_id] = occurrance['engagement_value'].mean()

        k = limit
        top_k = heapq.nlargest(k, content_id_to_score, key=lambda x: x["p_engage"])
        top_k_ids = list(map(lambda x: x["content_id"], top_k))
        return top_k_ids