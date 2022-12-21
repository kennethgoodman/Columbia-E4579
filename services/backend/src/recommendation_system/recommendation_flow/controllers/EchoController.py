from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)

from src.recommendation_system.recommendation_flow.controllers.ExampleController import (
    ExampleController,
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

import json

class EchoController(AbstractController):

    def __init__(self):
        self.filter = EchoFilter()
        self.candidate_generator = EchoGenerator()
        self.predictor = EchoModel()
        self.ranker = EchoRanker()

    def get_content_ids(self, user_id, limit=None, offset=None, seed=None, starting_point=None):

        # Check if the user exists. If not, use ExampleController (Most-popular ranking)
        recs_cb = json.load(open('src/echo_space/output/cg_cb_recs.json'))
        if str(user_id) not in recs_cb:
            return ExampleController().get_content_ids(user_id, limit, offset, seed, starting_point)

        candidates = self.candidate_generator.get_content_ids(user_id, limit=1000)
        filtered = self.filter.filter_ids(user_id, candidates)
        predictions = self.predictor.predict_probabilities(user_id, filtered)
        recs = self.ranker.rank_ids(predictions, limit, seed, starting_point)

        return recs
