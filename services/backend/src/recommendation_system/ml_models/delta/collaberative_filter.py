CollaberativeFilteredSimilarUsersGenerator.py
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.delta.UserBasedRecommender import UserBasedRecommender

class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        userBasedRecommender = UserBasedRecommender()
        content_ids, scores = userBasedRecommender.recommend_items(user_id, num_recommendations=limit + offset)
        return content_ids[offset:], scores[offset:]
    
    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
