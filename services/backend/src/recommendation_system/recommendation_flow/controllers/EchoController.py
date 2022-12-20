from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)

from src.recommendation_system.recommendation_flow.candidate_generators.EchoGenerator import (
    EchoGenerator,
)
from src.recommendation_system.recommendation_flow.filtering.EchoFilter import (
    EchoFilter,
)
from src.recommendation_system.recommendation_flow.model_prediction.EchoModel import (
    EchoModel,
)
from src.recommendation_system.recommendation_flow.ranking.EchoRanker import (
    EchoRanker,
)


class EchoController(AbstractController):

    def __init__(self):
        self.filter = EchoFilter()
        self.candidate_generator = EchoGenerator()
        self.predictor = EchoModel()
        self.ranker = EchoRanker()

    def get_content_ids(self, user_id, limit=None, offset=None, seed=None, starting_point=None):

        candidates = self.candidate_generator.get_content_ids(user_id, limit=1000)
        filtered = self.filter.filter(candidates)
        predictions = self.predictor.predict_probabilities(filtered, candidates)
        recs = self.ranker.rank_ids(predictions, limit, seed, starting_point)

        return recs

# controller = EchoController()
#
# recs = controller.get_content_ids(1, limit=20)
#
# print(recs)
