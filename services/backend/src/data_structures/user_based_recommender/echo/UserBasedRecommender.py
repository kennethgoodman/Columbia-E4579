
import pandas as pd
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement, EngagementType
from src import db
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedRecommender:

    _instance = None  

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserBasedRecommender, cls).__new__(cls)
            cls._instance.user_similarity_map = {}
            cls._instance.gather_data()
            cls._instance.compute_similarity()
        return cls._instance

    def gather_data(self):

        self.interactions = db.session.query(
            Engagement.content_id,
            Engagement.user_id,
            Engagement.engagement_type,
            Engagement.engagement_value
        ).all()

    def compute_similarity(self):


        user_item_dict = {}

        for interaction in self.interactions:
            content_id, user_id, engagement_type, engagement_value = interaction

            if user_id not in user_item_dict:
                user_item_dict[user_id] = {}
            user_item_dict[user_id][content_id] = user_item_dict[user_id].get(content_id, 0) + engagement_value

        users = list(user_item_dict.keys())
        contents = {content_id for user, content_data in user_item_dict.items() for content_id in content_data.keys()}
        matrix = []
        for user in users:

            row = [user_item_dict[user].get(content_id, 0) for content_id in contents]
            matrix.append(row)

        similarity_matrix = cosine_similarity(matrix)

        for idx, user in enumerate(users):
            self.user_similarity_map[user] = {}
            for jdx, other_user in enumerate(users):
                if user != other_user:  
                    self.user_similarity_map[user][other_user] = similarity_matrix[idx][jdx]


    def get_similar_users(self, user_id):

        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, num_recommendations=10):

        self.compute_similarity()
        similar_users = sorted(self.user_similarity_map.get(user_id, {}).items(), key=lambda x: x[1], reverse=True)

        recommended_content_ids = set()


        user_engaged_content = set([content_id for content_id, user1_id, _, _ in self.interactions if user1_id == user_id])

        for similar_user, _ in similar_users:

            similar_user_engaged_content = set([content_id for content_id, user_id, _, _ in self.interactions if user_id == similar_user])


            new_recommendations = similar_user_engaged_content - user_engaged_content
            recommended_content_ids.update(new_recommendations)


            if len(recommended_content_ids) >= num_recommendations:
                break


        return list(recommended_content_ids)[:num_recommendations]

