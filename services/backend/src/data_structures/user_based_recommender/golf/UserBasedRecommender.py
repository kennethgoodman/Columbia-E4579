import numpy as np
from src import db
from src.api.engagement.models import Engagement
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata

class UserBasedRecommender:
    _instance = None  # Singleton instance reference

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserBasedRecommender, cls).__new__(cls)
            cls._instance.user_similarity_map = {}
            cls._instance.user_id_to_index = {}  # Mapping of user_id to matrix index
            cls._instance.content_id_to_index = {}  # Mapping of content_id to matrix index
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
        # Create mappings of user_id and content_id to matrix indices
        user_ids = set(interaction.user_id for interaction in self.interactions)
        content_ids = set(interaction.content_id for interaction in self.interactions)
        self.user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        self.content_id_to_index = {content_id: index for index, content_id in enumerate(content_ids)}

    def compute_similarity(self):
        # Create a user-item interaction matrix (user rows, item columns)
        user_item_matrix = np.zeros((len(self.user_id_to_index), len(self.content_id_to_index)))

        # Populate the user-item matrix with engagement values from interactions
        for interaction in self.interactions:
            user_id = self.user_id_to_index.get(interaction.user_id)
            content_id = self.content_id_to_index.get(interaction.content_id)
            engagement_value = interaction.engagement_value
            if user_id is not None and content_id is not None:
                user_item_matrix[user_id, content_id] = engagement_value

        # Compute user similarity using cosine similarity
        user_norms = np.linalg.norm(user_item_matrix, axis=1)
        user_similarity_matrix = np.dot(user_item_matrix, user_item_matrix.T) / (
                np.outer(user_norms, user_norms) + 1e-10  # Add a small epsilon to avoid division by zero
        )

        self.user_similarity_map = {user_id: user_similarity_matrix[user_index] for user_id, user_index in self.user_id_to_index.items()}

    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        user_index = self.user_id_to_index.get(user_id)
        if user_index is None:
            return []  # User not found in the dataset
        user_similarity_vector = self.user_similarity_map.get(user_id, [])
        similar_users = []

        # Iterate through the similarity vector and select users with similarity above the threshold.
        for other_user_index, similarity_score in enumerate(user_similarity_vector):
            if similarity_score > 0:
                similar_users.append(
                    next((user_id for user_id, index in self.user_id_to_index.items() if index == other_user_index),
                         None))

        return similar_users

    def recommend_items(self, user_id, num_recommendations):
        # For a given user, fetch the list of similar users.
        similar_users = self.get_similar_users(user_id)

        # Initialize a dictionary to store content engagement by similar users
        content_engagement_by_similar_users = {}

        # Collect engagement information for content items by similar users
        for similar_user_id in similar_users:
            # Fetch engagement data for the similar user
            similar_user_engagement = db.session.query(
                Engagement.content_id,
                Engagement.engagement_value
            ).filter(Engagement.user_id == similar_user_id).all()

            # Store the engagement data in the dictionary
            for content_id, engagement_value in similar_user_engagement:
                if content_id not in content_engagement_by_similar_users:
                    content_engagement_by_similar_users[content_id] = 0
                content_engagement_by_similar_users[content_id] += engagement_value

        # Identify content items that the target user hasn't seen
        target_user_engagement = db.session.query(
            Engagement.content_id,
            Engagement.engagement_value
        ).filter(Engagement.user_id == user_id).all()

        seen_content_ids = set(content_id for content_id, _ in target_user_engagement)

        # Filter and rank the unseen content items based on engagement by similar users
        recommended_items = []
        for content_id, engagement_value in content_engagement_by_similar_users.items():
            if content_id not in seen_content_ids:
                recommended_items.append((content_id, engagement_value))

        # Sort the recommended items by engagement value in descending order
        recommended_items.sort(key=lambda x: x[1], reverse=True)

        # Select the top num_recommendations items as recommendations
        top_recommendations = recommended_items[:num_recommendations]

        # Extract content IDs from the recommendations
        recommended_content_ids = [content_id for content_id, _ in top_recommendations]

        return recommended_content_ids

