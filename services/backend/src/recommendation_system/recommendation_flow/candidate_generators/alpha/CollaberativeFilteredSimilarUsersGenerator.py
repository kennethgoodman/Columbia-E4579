
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.alpha.UserBasedRecommender import UserBasedRecommender

class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        # Use the recommender to get recommended content_ids for the user
        recommended_content_ids, content_ids_value = UserBasedRecommender().recommend_items(user_id, limit)
        return recommended_content_ids[offset:], content_ids_value[offset:] #placeholder for now

    
    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
