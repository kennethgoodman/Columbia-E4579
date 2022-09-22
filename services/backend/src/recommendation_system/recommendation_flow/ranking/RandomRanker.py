import heapq
import random

from .AbstractRanker import AbstractRanker


class RandomRanker(AbstractRanker):
    def rank_ids(self, limit, probabilities, seed, starting_point):
        k = limit
        top_k = heapq.nlargest(k, probabilities, key=lambda x: x["p_engage"])
        top_k_ids = list(map(lambda x: x["content_id"], top_k))
        if seed:
            random.seed(seed)
        return random.sample(top_k_ids, len(top_k_ids))  # shuffle
