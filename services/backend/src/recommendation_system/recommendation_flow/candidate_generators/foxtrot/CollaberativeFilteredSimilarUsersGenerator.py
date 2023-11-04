from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.foxtrot.UserBasedRecommender import UserBasedRecommender
from src.recommendation_system.ml_models.foxtrot.utils import get_tops


class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        if user_id:
            candidate_generator = UserBasedRecommender()
            candidate_generator.compute_similarity()
            res = candidate_generator.recommend_items(user_id, limit, offset)
            return res
        else:
            _, _, _, top_n_content = get_tops(None)
            res = top_n_content
            scores = [1.0] * len(res)
            return res, scores

    def _get_name(self):
        return "CollaberativeFilteredSimilarUsers"
