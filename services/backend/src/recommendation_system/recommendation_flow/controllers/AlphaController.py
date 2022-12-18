# from src.recommendation_system.recommendation_flow.candidate_generators.RandomGenerator import (
#     RandomGenerator,
# )
from src.recommendation_system.recommendation_flow.candidate_generators.AlphaGenerator import (
    AlphaGenerator,
)
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.AlphaFilter import (
    AlphaFilter,
)
# from src.recommendation_system.recommendation_flow.model_prediction.RandomModel import (
#     RandomModel,
# )
from src.recommendation_system.recommendation_flow.model_prediction.AlphaModel import (
    AlphaModel,
)
# from src.recommendation_system.recommendation_flow.ranking.RandomRanker import (
#     RandomRanker,
# )
from src.recommendation_system.recommendation_flow.ranking.AlphaRanker import (
    AlphaRanker,
)


class AlphaController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        print('limit from AlphaController get_content_ids():',limit)
        if seed <= 1:  # MySql seeds should be [0, # of rows] not [0, 1]
            seed *= 1000000
        candidates_limit = (
            limit * 10 * 10
        )  # 10% gets filtered out and take top 10% of rank
        candidates, scores = AlphaGenerator().get_content_ids(
            user_id, candidates_limit, offset, seed, starting_point
        )
        filtered_candidates = AlphaFilter().filter_ids(
            candidates, seed, starting_point
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
        # rank = RandomRanker().rank_ids(limit, predictions, seed, starting_point)
        rank = AlphaRanker().rank_ids(limit, predictions, seed, starting_point)
        print('rank',rank)
        return rank
