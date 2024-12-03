from src.recommendation_system.recommendation_flow.candidate_generators.ExampleGenerator import (
    ExampleGeneratorTextPct,
)
from src.recommendation_system.recommendation_flow.candidate_generators.RandomGenerator import (
    RandomGenerator
)
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.filtering.ExampleFilter import (
    ExampleFilterWithExploration,
)
from src.recommendation_system.recommendation_flow.model_prediction.ExampleModel import (
    ExampleModelWithForcedText,
)
from src.recommendation_system.recommendation_flow.ranking.ExampleRanker import (
    ExampleRanker,
)
from src.api.metrics.models import TeamName

class Fall2024Controller(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        candidates_limit = (
            limit * 10 * 10
        )  # 10% gets filtered out and take top 10% of rank
        candidatesEx, scoresEx = ExampleGeneratorTextPct().get_content_ids(
            TeamName.Example,
            user_id, candidates_limit // 2, offset, seed, starting_point
        )
        random_candidatesText, random_scoresText = RandomGenerator().get_content_ids(
            TeamName.Random,
            user_id, candidates_limit // 2, offset, seed, starting_point
        )
        candidates = candidatesEx + random_candidatesText
        scores = scoresEx + random_scoresText
        filtered_candidates = ExampleFilterWithExploration().filter_ids(
            TeamName.Example,  user_id, candidates, seed, starting_point
        )
        predictions = ExampleModelWithForcedText().predict_probabilities(
            TeamName.Example,
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
        rank = ExampleRanker().rank_ids(TeamName.Example, user_id, filtered_candidates, limit, predictions, seed, starting_point)
        return rank
