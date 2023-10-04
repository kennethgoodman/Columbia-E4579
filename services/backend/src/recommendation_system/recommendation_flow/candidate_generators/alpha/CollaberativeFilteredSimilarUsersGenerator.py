
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.alpha.UserBasedRecommender import UserBasedRecommender

class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        return [], []
    
    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
