
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.user_based_recommender.echo.UserBasedRecommender import UserBasedRecommender
from src.api.engagement.models import Engagement, EngagementType
from sqlalchemy.sql.expression import func
    

class CollaberativeFilteredSimilarUsersGenerator(AbstractGenerator):

    def _get_content_ids(self, _, limit, offset, _seed, starting_point):
        # if starting_point is None:
        recommendation_length = 200
        rec = UserBasedRecommender()
        rec.compute_similarity()
        lst  = rec.recommend_items(_, recommendation_length)
        lst2 = [0] * limit
        
        return lst, lst2
    
    def _get_name(self):
        return "CollaberativeFilteredSimilarUsersGenerator"



