
import pandas as pd
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement, EngagementType
from src import db
from sklearn.metrics.pairwise import cosine_similarity

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
        self.interactions = db.session.query(
            Engagement.content_id,
            Engagement.user_id,
            Engagement.engagement_type,
            Engagement.engagement_value
        ).all()

    def compute_similarity(self):
        # Compute the similarity between users.
        # For simplicity, we'll use cosine similarity as our metric.
        # Update self.user_similarity_map with the similarities.
        # 1. Create a user-item matrix using dictionaries
        user_item_dict = {}
        # filtered_interactions = [interaction for interaction in self.interactions if interaction.engagement_type == EngagementType.Like]
        for interaction in self.interactions:
            content_id, user_id, engagement_type, engagement_value = interaction

            if user_id not in user_item_dict:
                user_item_dict[user_id] = {}
            user_item_dict[user_id][content_id] = user_item_dict[user_id].get(content_id, 0) + engagement_value
        # print(user_item_dict)
        # Convert the dictionary to a list of lists for cosine similarity
        users = list(user_item_dict.keys())
        contents = {content_id for user, content_data in user_item_dict.items() for content_id in content_data.keys()}
        matrix = []
        for user in users:
            # print(user)
            row = [user_item_dict[user].get(content_id, 0) for content_id in contents]
            matrix.append(row)
        # print(matrix)
        similarity_matrix = cosine_similarity(matrix)
        # print(similarity_matrix)
        # Store results in user_similarity_map
        for idx, user in enumerate(users):
            self.user_similarity_map[user] = {}
            for jdx, other_user in enumerate(users):
                if user != other_user:  # Avoid comparing the user to themselves
                    self.user_similarity_map[user][other_user] = similarity_matrix[idx][jdx]
        # print(self.user_similarity_map)

    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, num_recommendations=10):
        # For a given user, fetch the list of similar users.
        # Recommend items engaged by those users, which the given user hasn't seen.
        # Fetch the list of similar users for the given user_id.
        self.compute_similarity()
        similar_users = sorted(self.user_similarity_map.get(user_id, {}).items(), key=lambda x: x[1], reverse=True)
        # print(self.user_similarity_map)
        # print(similar_users)
        # print(user_id)
        # Track recommended items.
        recommended_content_ids = set()

        # Items that the user has already engaged with.
        user_engaged_content = set([content_id for content_id, user1_id, _, _ in self.interactions if user1_id == user_id])
        # print(user_engaged_content)
        for similar_user, _ in similar_users:
            # Get items engaged by the similar user.
            similar_user_engaged_content = set([content_id for content_id, user_id, _, _ in self.interactions if user_id == similar_user])
            # print(similar_user_engaged_content)
            # Exclude items that the main user has already engaged with.
            new_recommendations = similar_user_engaged_content - user_engaged_content
            recommended_content_ids.update(new_recommendations)

            # If we have enough recommendations, break the loop.
            if len(recommended_content_ids) >= num_recommendations:
                break

        # print(recommended_content_ids)
        return list(recommended_content_ids)[:num_recommendations]

