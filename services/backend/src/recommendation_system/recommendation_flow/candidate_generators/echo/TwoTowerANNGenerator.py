
from src.recommendation_system.recommendation_flow.candidate_generators.AbstractGenerator import AbstractGenerator
from src.data_structures.approximate_nearest_neighbor.two_tower_ann import (
	get_ANN_recommendations_from_content,
	get_ANN_recommendations_from_user
)

class TwoTowerANNGenerator(AbstractGenerator):
    def _get_content_ids(self, user_id, limit, offset, _, starting_point):
        if starting_point.get("content_id", None) is not None:
            content_ids, scores = get_ANN_recommendations_from_content(starting_point["content_id"], "echo", limit + offset)
        else:
            content_ids, scores = get_ANN_recommendations_from_user(user_id, "echo", limit + offset)
        return content_ids[offset:], scores[offset:]

    def _get_name(self):
        return "TwoTowerANNGenerator"
