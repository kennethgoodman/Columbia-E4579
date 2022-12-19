from src.recommendation_system.recommendation_flow.candidate_generators.AlphaGenerator import (
    AlphaGenerator,
)
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.AlphaFilter import (
    AlphaFilter,
)
from src.recommendation_system.recommendation_flow.model_prediction.AlphaModel import (
    AlphaModel,
)
from src.recommendation_system.recommendation_flow.ranking.AlphaRanker import (
    AlphaRanker,
)


class AlphaController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        print('limit from AlphaController get_content_ids():',limit)
        candidates, scores = AlphaGenerator().get_content_ids(
            user_id
        )
        filtered_candidates = AlphaFilter().filter_ids(
            candidates
        )
        # predictions = RandomModel().predict_probabilities(
        predictions = AlphaModel().predict_probabilities(
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
        # print('predictions:',predictions)
        rank = AlphaRanker().rank_ids(limit, predictions, seed, starting_point)
        print('rank',rank)
        return rank
