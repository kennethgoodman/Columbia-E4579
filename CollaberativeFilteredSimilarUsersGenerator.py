
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.golf.UserBasedRecommender import UserBasedRecommender


class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, seed, starting_point):

        recommender = UserBasedRecommender()
        content_ids = recommender.recommend_items(user_id, limit)

        return content_ids, []

    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
