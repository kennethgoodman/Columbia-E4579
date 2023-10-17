
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pdb
from src.api.engagement.models import EngagementType, LikeDislike
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
        self.engagement_data = DataCollector().get_data_df()

    def compute_similarity(self, threshhold_percentile=70):
        like_data = self.engagement_data[self.engagement_data['engagement_type'] == EngagementType.Like]
        user_item_matrix = like_data.pivot_table(
            index='user_id',
            columns='content_id',
            values='engagement_value',
            fill_value=0,
            aggfunc='sum'
        )
        user_ids = user_item_matrix.index.tolist()
        for i, sim in enumerate(cosine_similarity(csr_matrix(user_item_matrix))):
            threshold = np.percentile(sim, threshhold_percentile)
            above_threshold = sim >= threshold
            self.user_similarity_map[user_ids[i]] = list(np.array(user_ids)[above_threshold])
            self.user_similarity_map[user_ids[i]].remove(user_ids[i])

    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, num_recommendations=500):
        # For a given user, fetch the list of similar users.
        # Recommend items engaged by those users, which the given user hasn't seen.
        similar_users = self.get_similar_users(user_id)
        seen_items = [interaction.content_id for interaction in self.interactions if interaction.user_id==user_id]
        recommend_items = set()
        for interaction in self.interactions:

            if(interaction.engagement_type==EngagementType.Like and interaction.engagement_value==1 and interaction.user_id in similar_users and interaction.content_id not in seen_items):
                recommend_items.add(interaction.content_id)
            if len(recommend_items) == num_recommendations:
                break
        return list(recommend_items)