from project.recommendation_flow.controllers.AbstractController import AbstractController
from project.recommendation_flow.candidate_generators.RandomGenerator import RandomGenerator
from project.recommendation_flow.filtering.RandomFilter import RandomFilter
from project.recommendation_flow.model_prediction.UntrainedModel import UntrainedModel
from project.recommendation_flow.ranking.RandomRanker import RandomRanker


class RandomController(AbstractController):
    def get_content_ids(self, user_id):
        candidates = RandomGenerator().get_content_ids()
        filtered_candidates = RandomFilter().filter_ids(candidates)
        predictions = UntrainedModel().predict_probabilities(filtered_candidates, user_id)
        rank = RandomRanker().rank_ids(predictions)
        return rank
