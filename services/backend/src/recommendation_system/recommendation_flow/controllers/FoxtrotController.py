from src.recommendation_system.recommendation_flow.candidate_generators.FoxtrotGenerator import (
    FoxtrotGenerator,
)
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.FoxtrotFilter import (
    FoxtrotFilter,
)
from src.recommendation_system.recommendation_flow.model_prediction.FoxtrotModel import (
    FoxtrotModel,
)
from src.recommendation_system.recommendation_flow.ranking.FoxtrotRanker import (
    FoxtrotRanker,
)


class StaticController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        seed = 0.25  # static random seed
        candidates_limit = (
            limit * 10 * 10
        )  # 10% gets filtered out and take top 10% of rank
        candidates, scores = FoxtrotGenerator().get_content_ids(
            user_id, candidates_limit, offset, seed, starting_point
        )
        filtered_candidates = FoxtrotFilter().filter_ids(
            candidates, seed, starting_point
        )
        predictions = FoxtrotModel().predict_probabilities(
            filtered_candidates,
            user_id,
            seed=seed,
            scores={
                content_id: {"score": score}
                for content_id, score in zip(candidates, scores or [])
            },
        )
        rank = FoxtrotRanker().rank_ids(limit, predictions, seed, starting_point)
        return rank
