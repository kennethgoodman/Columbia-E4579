from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db
import pickle
import os

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
        self.interactions = []
        self.min_val = db.session.query(
            func.min(Engagement.engagement_value)
            ).where(Engagement.engagement_type == "MillisecondsEngagedWith").one()[0]
        self.max_val = db.session.query(
            func.max(Engagement.engagement_value)
            ).where(Engagement.engagement_type == "MillisecondsEngagedWith").one()[0]

    def compute_similarity(self):
        # Compute the similarity between users.
        # For simplicity, we'll use cosine similarity as our metric.
        # Update self.user_similarity_map with the similarities.
        user_content_matrix = defaultdict(lambda: defaultdict(int))

        for interaction in self.interactions:
            content_id, user_id, engagement_type, value = interaction

            if engagement_type == "Like":
                if value == 1:
                    user_content_matrix[user_id][content_id] += 2
                else:
                    user_content_matrix[user_id][content_id] += -2
            else:
                # min-max scaling
                normalized_value = 0
                if value <= 6000:
                    normalized_value = (value - self.min_val) / (self.max_val - self.min_val)
                user_content_matrix[user_id][content_id] += normalized_value

        users = sorted(user_content_matrix.keys())
        contents = sorted({interaction[0] for interaction in self.interactions})
        matrix = np.zeros((len(users), len(contents)))

        for i, user in enumerate(users):
            for j, content in enumerate(contents):
                matrix[i][j] = user_content_matrix[user][content]

        similarities = cosine_similarity(matrix)

        for i, user in enumerate(users):
            self.user_similarity_map[user] = [(users[j], sim) for j, sim in enumerate(similarities[i]) if j != i and sim > 0]

        for user in users:
            self.user_similarity_map[user] = sorted(self.user_similarity_map.get(user, []), key=lambda x: x[1], reverse=True)



    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, limit, offset, num_recommendations=500):
        # For a given user, fetch the list of similar users.
        # Recommend items engaged by those users, which the given user hasn't seen.
        similar_users = self.user_similarity_map.get(user_id, [])
        content_scores = defaultdict(float)

        user_interacted_contents = {interaction[0] for interaction in self.interactions if
                                    interaction[1] == user_id}

        for similar_user, similarity_score in similar_users:
            similar_user_interactions = {interaction[0] for interaction in self.interactions if
                                         interaction[1] == similar_user}
            new_recommendations = similar_user_interactions - user_interacted_contents
            for content in new_recommendations:
                content_scores[content] = similarity_score

        # Sort the recommended contents by their scores in descending order and get the top num_recommendations
        sorted_recommendations = sorted(content_scores.keys(), key=lambda x: content_scores[x],
                                        reverse=True)[:num_recommendations]
        scores = [content_scores[content] for content in sorted_recommendations]

        if offset + limit < len(sorted_recommendations) and offset >= 0:
            return sorted_recommendations[offset:offset+limit], scores[offset:offset+limit]
        return sorted_recommendations, scores
