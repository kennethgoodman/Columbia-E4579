from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity 
import heapq
from src.data_structures.user_based_recommender.data_collector import DataCollector

class UserBasedRecommender:

    _instance = None  # Singleton instance reference

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserBasedRecommender, cls).__new__(cls)
            cls._instance.user_similarity_map = {}
            cls._instance.gather_data()
            cls._instance.compute_similarity()
        return cls._instance

    def gather_data(self):
        # Connect to the database and fetch user-content engagement.
        self.interactions_df = DataCollector().get_data_df()
        #we only get 100000 data but can get more/all

    def aggregate_engagement(self, group):
        #summing millisecond engagement values
        millisecond_engagement_sum = group.loc[group['engagement_type'] != 'Like', 'engagement_value'].sum()
        
        # Counting likes and dislikes
        #likes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == 1)].shape[0]
        #dislikes_count = group.loc[(group['engagement_type'] == 'Like') & (group['engagement_value'] == -1)].shape[0]
        
        return pd.Series({
            'millisecond_engagement_sum': millisecond_engagement_sum,
            #'likes_count': likes_count,
            #'dislikes_count': dislikes_count
        })    
        
    def compute_similarity(self):
        # Compute the similarity between users.
        # For simplicity, we'll use cosine similarity as our metric.
        # Update self.user_similarity_map with the similarities.
        
        TOP_CONTENT = 251
        interactions_df = self.interactions_df
        # Compute top N content pieces based on engagement count
        top_n_content = interactions_df.groupby('content_id')['engagement_value'].count().nlargest(TOP_CONTENT).index.tolist()
        #initiliaze  user_vector_dict
        user_vector_dict = defaultdict(lambda: {
            'millisecond_engaged_vector': np.zeros(len(top_n_content))
            #'like_vector': np.zeros(len(top_n_content)),
            #'dislike_vector': np.zeros(len(top_n_content))
        })        
        engagement_aggregate = interactions_df[interactions_df['content_id'].isin(top_n_content)].groupby(['user_id', 'content_id']).apply(self.aggregate_engagement).reset_index()
        # Now, populate your user_vector_dict, mapping user_id to list of engagement_sums for top contents
        #user_vector_dict={user_id: [engagement_sum for content i, i=[0, 250]]}
        for _, row in engagement_aggregate.iterrows():
            user_id = row['user_id']
            content_id = row['content_id']
            idx = top_n_content.index(content_id)
            
            user_vector_dict[user_id]['millisecond_engaged_vector'][idx] = row['millisecond_engagement_sum']
            #user_vector_dict[user_id]['like_vector'][idx] = row['likes_count']
            #user_vector_dict[user_id]['dislike_vector'][idx] = row['dislikes_count']
        
        user_vector_df = pd.DataFrame.from_dict(user_vector_dict, orient='index')
        del user_vector_dict
        millisecond_columns = [f"ms_engaged_{i}" for i in range(TOP_CONTENT)]
        #like_columns = [f"like_vector_{i}" for i in range(TOP_CONTENT)]
        #dislike_columns = [f"dislike_vector_{i}" for i in range(TOP_CONTENT)]
        user_vector_df[millisecond_columns] = pd.DataFrame(user_vector_df['millisecond_engaged_vector'].tolist(), index=user_vector_df.index)
        #user_vector_df[like_columns] = pd.DataFrame(user_vector_df['like_vector'].tolist(), index=user_vector_df.index)
        #user_vector_df[dislike_columns] = pd.DataFrame(user_vector_df['dislike_vector'].tolist(), index=user_vector_df.index)

        # Drop the original vector columns
        user_vector_df.drop(['millisecond_engaged_vector'], axis=1, inplace=True)
        user_vector_df = user_vector_df.reset_index().rename(columns={'index': 'user_id'})
        # Create overall features
        user_columns = (
            [f'ms_engaged_{i}' for i in range(TOP_CONTENT)] 
            #+ [f'like_vector_{i}' for i in range(TOP_CONTENT)]
            #+ [f'dislike_vector_{i}' for i in range(TOP_CONTENT)]
        )
        user_features = user_vector_df[user_columns]
        # mapping user_id to its index in user features
        user_idx_dict = {idx: row['user_id'] for idx, row in user_vector_df.iterrows()}

        #user_features_tensor = torch.FloatTensor(user_features.values)

        user_similarity = cosine_similarity(user_features)
        SIMILAR_USERS = 20
        for user_idx, similarities in enumerate(user_similarity):
            user_id = user_idx_dict[user_idx]
            top_similar = heapq.nlargest(SIMILAR_USERS + 1, enumerate(similarities), key=lambda x: x[1])
            self.user_similarity_map[user_id] = [user_idx_dict[idx] for idx, _ in top_similar if idx != user_idx]



    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, num_recommendations=10):
        # For a given user, fetch the list of similar users.
        # Recommend items engaged by those users, which the given user hasn't seen.
        
        #can store this df in compute similarity as a class field
        interactions_df = self.interactions_df
        similar_users = self.get_similar_users(user_id)
        content_dict = {}
        seen_content_ids = interactions_df[interactions_df["user_id"] == user_id]["content_id"].unique()

        for similar_user_id in similar_users:
            content_id_list = interactions_df[interactions_df["user_id"] == similar_user_id]["content_id"].unique()
            for content_id in content_id_list:
                if content_id not in seen_content_ids:
                    if content_id in content_dict:
                        content_dict[content_id] += 1
                    else:
                        content_dict[content_id] = 1

        #currently, we order items based on the number of similar users who have engaged with them
        top_items_tuples = heapq.nlargest(num_recommendations, content_dict.items(), key=lambda x:x[1])
        top_items = [item[0] for item in top_items_tuples]
        top_item_scores = [item[1] for item in top_items_tuples]

        #maybe add a solution in case we couldn't generate enough unseen content => increase top users limit

        return top_items, top_item_scores
        
