from src.recommendation_system.recommendation_flow.candidate_generators.CharlieGenerator import (
    CharlieGenerator,
)
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.CharlieFilter1 import (
    DislikeRatioFilter,
)
from src.recommendation_system.recommendation_flow.model_prediction.CharlieModel import (
    CharlieModel,
)
from src.recommendation_system.recommendation_flow.ranking.CharlieRanking import (
    CharlieRanker,
)


class CharlieController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        candidates_limit = (
            limit * 10 * 10
        )  # 10% gets filtered out and take top 10% of rank
        candidates, scores = CharlieGenerator().get_content_ids(
            user_id, candidates_limit, offset, seed, starting_point
        )
        filtered_candidates = DislikeRatioFilter().filter_ids(
            candidates, seed, starting_point
        )
        predictions = CharlieModel().predict_probabilities(
            filtered_candidates,
            user_id,
            seed=seed,
            scores={
                content_id: {"score": score}
                for content_id, score in zip(candidates, scores)
            }
            if scores is not None
            else {},
        )
        rank = CharlieRanker().rank_ids(limit, predictions, seed, starting_point)
        return rank
