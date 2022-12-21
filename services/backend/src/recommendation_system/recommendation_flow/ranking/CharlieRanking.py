import heapq
import random

from .AbstractRanker import AbstractRanker


class RandomRanker(AbstractRanker):
    def rank_ids(self, limit, probabilities, seed, starting_point):
        k = limit
        double = 2*k
        top_2k = heapq.nlargest(double, probabilities, key=lambda x: x["score"])
        top_2k_ids = list(map(lambda x: x["content_id"], top_2k))
        if seed:
            random.seed(seed)
        return random.sample(top_2k_ids, k)  