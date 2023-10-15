
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pdb
from src.api.engagement.models import EngagementType, LikeDislike

class UserBasedRecommender:

    _instance = None  # Singleton instance reference

    def __new__(cls):
        if cls._instance is None:
            #print("init function")
            cls._instance = super(UserBasedRecommender, cls).__new__(cls)
            cls._instance.user_similarity_map = {}
            cls._instance.gather_data()
            cls._instance.compute_similarity()
        return cls._instance

    def gather_data(self):
        #print("gather data function")
        # Connect to the database and fetch user-content engagement.
        self.interactions = db.session.query(
            Engagement.content_id,
            Engagement.user_id,
            Engagement.engagement_type,
            Engagement.engagement_value
        ).all()
        #print("inside gather data", self.interactions)


    def compute_similarity(self, threshhold_percentile=70):
        # Compute the similarity between users.
        # For simplicity, we'll use cosine similarity as our metric.
        # Update self.user_similarity_map with the similarities.
        user_ids = list(set(interaction.user_id for interaction in self.interactions))
        #print("0", user_ids[:10])
        #print("A", self.interactions[:10])
        #print("B", list(set(interaction.content_id for interaction in self.interactions))[:10])
        item_ids = list(set(interaction.content_id for interaction in self.interactions))

        n_users = len(user_ids)
        n_items = len(item_ids)
        
        user_item_matrix = pd.DataFrame(np.zeros((n_users, n_items)),columns=item_ids,index=user_ids)
        for interaction in self.interactions:
            #print('--------', interaction.engagement_type)
            if interaction.engagement_type == EngagementType.Like:
                #print('##########')
                user_item_matrix.loc[interaction.user_id,interaction.content_id] += interaction.engagement_value
        
        r = cosine_similarity(np.array(user_item_matrix))
        for i, sim in enumerate(r):
            threshhold = np.percentile(sim,threshhold_percentile)
            self.user_similarity_map[user_ids[i]] = np.array(user_ids)[sim>=threshhold].tolist()
            self.user_similarity_map[user_ids[i]].remove(user_ids[i])


        #print('end of compute similaraity')
        #print(self.user_similarity_map)


    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, num_recommendations=500):
        # For a given user, fetch the list of similar users.
        # Recommend items engaged by those users, which the given user hasn't seen.
        similar_users = self.get_similar_users(user_id)
        #print("similar_users", similar_users)
        seen_items = [interaction.content_id for interaction in self.interactions if interaction.user_id==user_id]
        #print("seen_itesm", seen_items)
        recommend_items = set()
        for interaction in self.interactions:
            #print('Z', interaction)
            #print('Y', interaction.content_id)
            #print(interaction.engagement_type)
            # print(interaction.engagement_value)
            # print(interaction.user_id)
            # print(seen_items)
            if(interaction.engagement_type==EngagementType.Like and interaction.engagement_value==1 and interaction.user_id in similar_users and interaction.content_id not in seen_items):
                #print('inside if condn')
                recommend_items.add(interaction.content_id)
            if len(recommend_items) == num_recommendations:
                break
        #print("recommend_items", recommend_items)
        return list(recommend_items)