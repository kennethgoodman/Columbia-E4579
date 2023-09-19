
from sqlalchemy.sql.expression import func
from src.api.content.models import Content, GeneratedContentMetadata
from src.api.engagement.models import Engagement
from src import db

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
        ).limit(5).all()

    def compute_similarity(self):
        # Compute the similarity between users.
        # For simplicity, we'll use cosine similarity as our metric.
        # Update self.user_similarity_map with the similarities.
        pass

    def get_similar_users(self, user_id):
        # Fetch the list of similar users for a given user_id from the map.
        return self.user_similarity_map.get(user_id, [])

    def recommend_items(self, user_id, num_recommendations=10):
        # For a given user, fetch the list of similar users.
        # Recommend items engaged by those users, which the given user hasn't seen.
        pass

# This will ensure that there's only one instance of UserBasedRecommender throughout the application.
recommender = UserBasedRecommender()
