from src.recommendation_system.recommendation_flow.candidate_generators.DeltaGenerator import (
    UserPreferenceGenerator, CFGenerator, PopularCategoryGenerator
)

from src.recommendation_system.recommendation_flow.candidate_generators.RandomGenerator import (
    RandomGenerator
)

from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)

from src.recommendation_system.recommendation_flow.filtering.DeltaFilter import (
    QualityFilter
)

from src.recommendation_system.recommendation_flow.model_prediction.RandomModel import (
    RandomModel
)

from src.recommendation_system.recommendation_flow.model_prediction.DeltaModel import (
    RuleBasedModel,
)

from src.recommendation_system.recommendation_flow.ranking.DeltaRanker import (
    RuleBasedRanker,
)

import time

class DeltaController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        candidates_limit = (
            limit * 10 * 10
        )

        print(f"CONTROLLER: user id: {user_id}")

        start_time = time.time()

        if user_id == 0:
            candidates, scores = RandomGenerator().get_content_ids(
                user_id, candidates_limit, offset, seed, starting_point
            )
        else:
            candidates_1, scores_1 = UserPreferenceGenerator().get_content_ids(
                user_id, candidates_limit, offset, seed, starting_point
            )

            print(f"CONTROLLER: Finished CG1 in {time.time() - start_time}s.")
            start_time = time.time()

            candidates_2, scores_2 = CFGenerator().get_content_ids(
                user_id, candidates_limit, offset, seed, starting_point
            )

            print(f"CONTROLLER: Finished CG2 in {time.time() - start_time}s.")
            start_time = time.time()

            candidates_3, scores_3 = PopularCategoryGenerator().get_content_ids(
                user_id, candidates_limit, offset, seed, starting_point
            )

            print(f"CONTROLLER: Finished CG3 in {time.time() - start_time}s.")
            start_time = time.time()

            print(f"CONTROLLER: num candidates (user preference): {len(candidates_1)}")
            print(f"CONTROLLER: num candidates (cf) : {len(candidates_2)}")
            print(f"CONTROLLER: num candidates (popular) : {len(candidates_3)}")

            candidates = candidates_1 + candidates_2 + candidates_3

        filtered_candidates_scores = QualityFilter().filter_ids(
            candidates, seed, starting_point, user_id
        )

        filtered_candidates = filtered_candidates_scores.keys()

        print(f"CONTROLLER: Finished filtering in {time.time() - start_time}s.")
        start_time = time.time()

        if user_id == 0:
            predictor_model = RandomModel()
        else:
            predictor_model = RuleBasedModel()

        predictions = RuleBasedModel().predict_probabilities(
            filtered_candidates,
            user_id,
            seed=seed,
            scores={
                content_id: {"score": filtered_candidates_scores[content_id]}
                for content_id in filtered_candidates_scores
            }
        )

        print(f"CONTROLLER: Finished prediction in {time.time() - start_time}s.")
        start_time = time.time()

        rank = RuleBasedRanker().rank_ids(limit, predictions, seed, starting_point)

        print(f"CONTROLLER: Finished ranking in {time.time() - start_time}s.")
        start_time = time.time()

        return list(set(rank))
