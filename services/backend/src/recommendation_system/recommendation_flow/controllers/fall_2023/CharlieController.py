
from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.candidate_generators.charlie.TwoTowerANNGenerator import TwoTowerANNGenerator
from src.recommendation_system.recommendation_flow.candidate_generators.charlie.CollaberativeFilteredSimilarUsersGenerator import CollaberativeFilteredSimilarUsersGenerator
from src.recommendation_system.recommendation_flow.candidate_generators.charlie.YourChoiceGenerator import YourChoiceGenerator
from src.recommendation_system.recommendation_flow.shared_data_objects.data_collector import DataCollector
from src.recommendation_system.recommendation_flow.filtering.fall_2023.CharlieFilter import (
    CharlieFilter,
)
from src.recommendation_system.recommendation_flow.model_prediction.RandomModel import (
    RandomModel,
)
from src.recommendation_system.recommendation_flow.model_prediction.fall_2023.charlie.CharlieModel import (
    CharlieFeatureGeneration, CharlieModel,
)
from src.recommendation_system.recommendation_flow.ranking.fall_2023.CharlieRanker import (
    CharlieRanker,
)
from src.api.metrics.models import TeamName

class CharlieController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        if seed <= 1:  # MySql seeds should be [0, # of rows] not [0, 1]
            seed *= 1000000
        candidate_limit = 100
        candidates, scores = [], []
        generators = []
        if starting_point.get("twoTower", False):
            generators.append(TwoTowerANNGenerator)
        if starting_point.get("collabFilter", False):
            generators.append(CollaberativeFilteredSimilarUsersGenerator)
        if starting_point.get("yourChoice", False):
            generators.append(YourChoiceGenerator)
        for gen in generators:
           cur_candidates, cur_scores = gen().get_content_ids(
               TeamName.Charlie_F2023,
               user_id, candidate_limit, offset, seed, starting_point
           )
           candidates += cur_candidates
           scores += cur_scores
        dc = DataCollector()
        dc.gather_data(user_id, candidates)
        filtered_candidates = CharlieFilter().filter_ids(
            TeamName.Charlie_F2023,
            user_id, candidates, seed, starting_point, dc=dc
        )
        charlieFG = CharlieFeatureGeneration(dc, filtered_candidates)
        if starting_point.get('randomPredictions'):
            model = RandomModel()
        else:
            model = CharlieModel()
        predictions = model.predict_probabilities(
            TeamName.Charlie_F2023,
            filtered_candidates,
            user_id,
            seed=seed,
            scores={
                content_id: {"score": score}
                for content_id, score in zip(candidates, scores)
            }
            if scores is not None
            else {},
            fg=charlieFG,
        )
        rank = CharlieRanker().rank_ids(
            TeamName.Charlie_F2023,
            user_id, filtered_candidates, limit, predictions, seed, starting_point, charlieFG.X_all
        )
        return rank
