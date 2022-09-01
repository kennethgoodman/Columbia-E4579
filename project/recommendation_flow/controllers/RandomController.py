from project.recommendation_flow.controllers.AbstractController import AbstractController
from project.recommendation_flow.candidate_generators.RandomGenerator import RandomGenerator
from project.recommendation_flow.filtering.RandomFilter import RandomFilter
from project.recommendation_flow.model_prediction.RandomModel import RandomModel
from project.recommendation_flow.ranking.RandomRanker import RandomRanker


class RandomController(AbstractController):
    def get_content_ids(self, user_id, limit, offset):
        candidates = RandomGenerator().get_content_ids(limit, offset)
        filtered_candidates = RandomFilter().filter_ids(candidates)
        predictions = RandomModel().predict_probabilities(filtered_candidates, user_id)
        rank = RandomRanker().rank_ids(predictions)
        return rank
