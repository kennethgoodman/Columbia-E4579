from src.recommendation_system.recommendation_flow.candidate_generators.EchoGenerator import (
    EchoGenerator,
)
# from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
#     AbstractController,
# )
# from src.recommendation_system.recommendation_flow.filtering.ExampleFilter import (
#     ExampleFilter,
# )
# from src.recommendation_system.recommendation_flow.model_prediction.ExampleModel import (
#     ExampleModel,
# )
# from src.recommendation_system.recommendation_flow.ranking.RandomRanker import (
#     RandomRanker,
# )


class EchoController:
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):


        candidates = EchoGenerator().get_content_ids(user_id, limit=100)

        #rank = RandomRanker().rank_ids(limit, predictions, seed, starting_point)
        return candidates
