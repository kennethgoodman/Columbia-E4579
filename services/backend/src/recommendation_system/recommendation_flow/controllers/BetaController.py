from src.recommendation_system.recommendation_flow.candidate_generators.BetaGenerator import (
    BetaGenerator,
)

from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.BetaFilter import (
    BetaFilter,
)

from src.recommendation_system.recommendation_flow.model_prediction.BetaModel import (
    BetaModel,
)

from src.recommendation_system.recommendation_flow.ranking.BetaRanker import (
    BetaRanker,
)


class BetaController(AbstractController):
    shown=[]

    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        print(self.shown)
        print(limit, offset)
        candidates_limit = (
            limit * 10 * 10
        )  # 10% gets filtered out and take top 10% of rank
        candidates, scores = BetaGenerator().get_content_ids(
            user_id, candidates_limit, offset, seed, starting_point
        )
        filtered_candidates = BetaFilter().filter_ids(
            candidates, user_id, seed, starting_point, BetaController.shown
        )
        predictions = BetaModel().predict_probabilities(
            filtered_candidates,
            user_id,
            seed = seed,
            scores = {
                content_id: {"score": score}
                for content_id, score in zip(candidates, scores)
            }
            if scores is not None
            else {},
        )
        rank = BetaRanker().rank_ids(limit, predictions, seed, starting_point)
        BetaController.shown.extend(rank)

        #print(BetaController.shown)
        #print('controller final 1-10th ids:', rank[:])
        
        return rank
