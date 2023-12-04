import heapq
import random

from .AbstractRanker import AbstractRanker


class RandomRanker(AbstractRanker):
    def _rank_ids(self, user_id, content_ids, limit, probabilities, seed, starting_point, X=None):
        if seed:
            random.seed(seed)
        return random.sample(content_ids, min(limit, len(content_ids)))  # shuffle

    def _get_name(self):
        return "RandomRanker"
