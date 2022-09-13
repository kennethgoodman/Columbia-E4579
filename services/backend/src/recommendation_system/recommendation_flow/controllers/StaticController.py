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


class StaticController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, _):
        seed = 0.25  # static random seed
        print("get_content_ids", user_id, limit, offset, seed)
        candidates = RandomGenerator().get_content_ids(limit, offset, seed)
        print("candidates", candidates)
        filtered_candidates = RandomFilter().filter_ids(candidates, seed)
        predictions = RandomModel().predict_probabilities(
            filtered_candidates, user_id, seed
        )
        rank = RandomRanker().rank_ids(predictions, seed)
        return rank
