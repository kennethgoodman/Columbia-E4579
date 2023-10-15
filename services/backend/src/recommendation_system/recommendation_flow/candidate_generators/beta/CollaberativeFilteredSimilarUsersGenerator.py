
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.beta.UserBasedRecommender import UserBasedRecommender
import pdb


class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        r = UserBasedRecommender()
        print('r initial completed')
        re = r.recommend_items(user_id, num_recommendations=500)
        return re[offset:offset+limit], [0]*limit
    
    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
