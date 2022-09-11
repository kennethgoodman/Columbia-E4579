from src.recommendation_system.recommendation_flow.candidate_generators.RandomGenerator import (
    RandomGenerator,
)
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.RandomFilter import (
    RandomFilter,
)
from src.recommendation_system.recommendation_flow.model_prediction.RandomModel import (
    RandomModel,
)
from src.recommendation_system.recommendation_flow.ranking.RandomRanker import (
    RandomRanker,
)


class RandomController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed):
        candidates = RandomGenerator().get_content_ids(limit, offset, seed)
        filtered_candidates = RandomFilter().filter_ids(candidates)
        predictions = RandomModel().predict_probabilities(filtered_candidates, user_id)
        rank = RandomRanker().rank_ids(predictions)
        return rank
