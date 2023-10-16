
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.alpha.UserBasedRecommender import UserBasedRecommender

class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        # Use the recommender to get recommended content_ids for the user
        recommended_content_ids, content_ids_value = UserBasedRecommender().recommend_items(user_id, limit)

        # Handle offset if necessary. For example, if offset is 5, then skip the first 5 recommendations.
        if offset:
            recommended_content_ids = recommended_content_ids[offset:]
            content_ids_value = content_ids_value[offset:]

        # Handle starting_point if necessary. This depends on its intended functionality.
        # Assuming starting_point is an ID after which we start the recommendations:
        if starting_point in recommended_content_ids:
            index = recommended_content_ids.index(starting_point)
            recommended_content_ids = recommended_content_ids[index+1:]
            content_ids_value = content_ids_value[index+1:]

        return recommended_content_ids,content_ids_value #placeholder for now

    
    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
