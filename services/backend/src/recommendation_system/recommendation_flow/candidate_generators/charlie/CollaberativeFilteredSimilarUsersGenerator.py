import pandas as pd
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.charlie.UserBasedRecommender import UserBasedRecommender


class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _seed, starting_point):
        # Get recommendations based on user-based collaborative filtering
        ubr = UserBasedRecommender()
        ubr.compute_similarity()
        recommended_content = ubr.recommend_items(user_id, 2 * limit)
        content_ids = []
        scores = []
        index = 0
        while index < 2 * limit:
            content_ids.append(recommended_content[index][0])
            if recommended_content[index][1] == 1:
                scores.append(1 + 3000 / recommended_content[index + 1][1])
                index += 2
            else:
                scores.append(recommended_content[index][1] / 30001)
                index += 1
        temp = pd.Series(scores, index=content_ids).sort_values(ascending=False)
        content_ids = temp.index.tolist()
        scores = temp.tolist()
        return content_ids[:limit], scores[:limit]

    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"
